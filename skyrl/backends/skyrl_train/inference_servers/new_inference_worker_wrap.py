"""
vLLM Worker Extension for native weight sync with chunked transfer support.

This module provides NewInferenceWorkerWrap, a vLLM worker extension that
enables chunked weight updates from training to inference using the
start/update/finish lifecycle:

    start_weight_update   ->  one or more update_weights_ipc  ->  finish_weight_update

This separates the layerwise reload initialization/finalization from individual
chunk transfers, allowing weights to be sent in bounded-memory chunks rather
than all at once.

Used only with the new inference path (_SKYRL_USE_NEW_INFERENCE=1).

TODO: Once https://github.com/vllm-project/vllm/pull/39212 lands, vLLM will
natively support start_weight_update / update_weights / finish_weight_update
on GPUWorker with dedicated HTTP endpoints. At that point this worker extension
can be removed and SkyRL can call the native endpoints directly instead of
routing through /collective_rpc.

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls \
        skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap
"""

import torch

# Workaround for a vLLM layerwise-reload corruption affecting NemotronH/Mamba.
# MambaMixer2 registers `conv_weights` as a non-persistent buffer that is a
# view of `self.conv1d.weight.data` (shared storage). vLLM's reload code path
# (model_executor/model_loader/reload/layerwise.py) materializes the buffer
# into a fresh uninitialized GPU tensor and then runs
# `kernel_conv_weights.data.copy_(fresh)` in `_copy_and_restore_kernel_tensors`.
# Because the kernel buffer shares storage with `conv1d.weight.data`, this
# writes garbage (NaN-bit-pattern bytes in bf16) into the conv1d weight,
# corrupting all 23 Mamba layers after every weight sync.
#
# Adding "conv_weights" to vLLM's SKIP_TENSORS makes capture/restore/materialize
# skip the buffer entirely, so the view stays intact and conv1d.weight is
# preserved. Must be applied before `record_metadata_for_reloading` runs at
# model construction; this module is imported by vLLM via
# --worker-extension-cls before model init, so the import-time patch is
# correctly ordered.
# Remove this pending https://github.com/vllm-project/vllm/pull/42481 which should
# be included in vLLM 0.21.0
try:
    from vllm.model_executor.model_loader.reload.meta import (
        SKIP_TENSORS as _VLLM_SKIP_TENSORS,
    )

    _VLLM_SKIP_TENSORS.add("conv_weights")
except ImportError:
    pass

# Register the SkyRL sparse-bf16 delta engine with vLLM's weight-transfer factory. This
# module is imported (via --worker-extension-cls) before model init, so the "delta" backend
# is registered by the time vLLM's factory builds the engine from
# WeightTransferConfig(backend="delta"). Guarded so a re-import or a vLLM build without the
# factory does not raise.
try:
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    if "delta" not in WeightTransferEngineFactory._registry:
        WeightTransferEngineFactory.register_engine(
            "delta",
            "skyrl.backends.skyrl_train.weight_sync.delta_engine",
            "DeltaWeightTransferEngine",
        )
except Exception:
    pass


# Capture the pre-quantization bf16 master at model load (delta backend only).
#
# A quantized inference model has no bf16 master after load: each weight is cast to
# fp8/kernel format and bf16<-fp8 is lossy. Delta sync's shard shadow needs that bf16 master
# to reconstruct the merge base locally, which is what lets the disk transport skip shipping
# the initial full-weight "seed". The bf16 value of a weight exists for one instant per layer
# -- right after its weight loaders run, right before its `process_weights_after_loading`
# quantizes it -- so we snapshot it there.
#
# vLLM has two such bf16->kernel boundaries; we wrap both so the capture generalizes across
# quant strategies without any per-quant-method knowledge:
#
#   (1) meta-device "online" quant (`quantization=fp8` etc.; `uses_meta_device=True`): each
#       layer is materialized + loaded + quantized inside
#       `reload.layerwise._layerwise_process`, *before* the model-level
#       `process_weights_after_loading` runs. We wrap `_layerwise_process` and, on the initial
#       load only, snapshot the layer's bf16 weights at the entry of its quant method's
#       `process_weights_after_loading`.
#   (2) non-meta "quantize in postprocess" (and plain bf16): weights stay bf16 in their real
#       params until the model-level `base_loader.process_weights_after_loading`. We wrap that
#       and snapshot model-wide before it casts.
#
# Both feed the same module-identity-keyed stash in `delta_utils`. Import-time (via
# --worker-extension-cls) so the wraps are in place before the model loads; gated to the
# delta backend so non-delta runs pay nothing.
def _skyrl_delta_backend_active() -> bool:
    """True iff the current vLLM config selects the delta weight-transfer backend."""
    try:
        from vllm.config import get_current_vllm_config

        wt_cfg = get_current_vllm_config().weight_transfer_config
        return wt_cfg is not None and getattr(wt_cfg, "backend", None) == "delta"
    except Exception:
        return False


# Seam (2): non-meta quantize-in-postprocess + bf16, via the model-level call. We patch
# `base_loader.process_weights_after_loading` (not the `utils` source): `DefaultModelLoader`
# inherits `base_loader.load_model` and hits this binding. Loaders that override `load_model`
# (gguf/tensorizer/modelexpress) hold their own bindings, but those are pre-quantized/
# specialized formats with no transient bf16 master to capture -- intentionally out of scope.
try:
    from vllm.model_executor.model_loader import base_loader as _skyrl_base_loader

    _skyrl_orig_process_weights = _skyrl_base_loader.process_weights_after_loading

    def _skyrl_capturing_process_weights(model, *args, **kwargs):
        try:
            if _skyrl_delta_backend_active():
                from skyrl.backends.skyrl_train.weight_sync.delta_utils import (
                    capture_model_prequant_bf16,
                )

                capture_model_prequant_bf16(model)
        except Exception:
            # Never let capture break model load; bf16/online-quant params are still covered
            # by the live snapshot / seam (1), respectively.
            pass
        return _skyrl_orig_process_weights(model, *args, **kwargs)

    _skyrl_base_loader.process_weights_after_loading = _skyrl_capturing_process_weights
except Exception:
    pass

# Seam (1): meta-device online quant, via the per-layer `_layerwise_process`. The bf16 weights
# live in the layer only between its weight loaders and its `quant_method
# .process_weights_after_loading` (both inside `_layerwise_process`), so we intercept that one
# call: on the initial load we temporarily wrap the layer's quant-method hook to snapshot the
# bf16 weights first. Reloads (where `info.kernel_tensors` is set) are skipped -- the shadow's
# own copy_ seam captures bf16 there.
try:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase as _SkyrlQuantizeMethodBase,
    )
    from vllm.model_executor.model_loader.reload import layerwise as _skyrl_layerwise

    _skyrl_orig_layerwise_process = _skyrl_layerwise._layerwise_process

    def _skyrl_capturing_layerwise_process(layer, info):
        is_initial_load = getattr(info, "kernel_tensors", None) is None
        quant_method = getattr(layer, "quant_method", None)
        if is_initial_load and isinstance(quant_method, _SkyrlQuantizeMethodBase) and _skyrl_delta_backend_active():
            # Bound hook captured before the swap; calling it auto-passes `self`.
            orig_pwal = quant_method.process_weights_after_loading

            def _capture_then_process(captured_layer, *a, **k):
                try:
                    from skyrl.backends.skyrl_train.weight_sync.delta_utils import (
                        capture_layer_prequant_bf16,
                    )

                    capture_layer_prequant_bf16(captured_layer)
                except Exception:
                    pass
                return orig_pwal(captured_layer, *a, **k)

            # Instance attribute shadows the class method for this one process call.
            quant_method.process_weights_after_loading = _capture_then_process
            try:
                return _skyrl_orig_layerwise_process(layer, info)
            finally:
                try:
                    del quant_method.process_weights_after_loading
                except Exception:
                    quant_method.process_weights_after_loading = orig_pwal
        return _skyrl_orig_layerwise_process(layer, info)

    _skyrl_layerwise._layerwise_process = _skyrl_capturing_layerwise_process
except Exception:
    pass

VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS = f"{__name__}.NewInferenceWorkerWrap"


class NewInferenceWorkerWrap:
    """
    vLLM worker extension for chunked weight sync (new inference path).

    Provides a three-phase weight update protocol via collective_rpc:
        1. start_weight_update: Prepare model for receiving weights
        2. update_weights_ipc: Receive and load one chunk of weights
        3. finish_weight_update: Finalize the model after all chunks

    Attributes accessed from the host GPUWorker (via mixin inheritance):
        self.weight_transfer_engine
        self.model_runner
        self.model_config
        self.device
    """

    def skyrl_start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """
        Prepare the model for a new weight update.

        For checkpoint-format weights, initializes the layerwise reload
        machinery which moves layers to meta device and wraps weight loaders
        to defer processing until all weights for each layer are loaded.

        Must be called before any update_weights_ipc calls.

        Args:
            is_checkpoint_format: True if incoming weights are in checkpoint
                format (need layerwise processing). False if weights are
                already in kernel format (direct copy).
        """
        if getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError(
                "start_weight_update called while a weight update is "
                "already active. Call finish_weight_update first."
            )

        if is_checkpoint_format:
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                initialize_layerwise_reload,
            )

            model = self.model_runner.model
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                initialize_layerwise_reload(model)

                # --- Delta shard-shadow integration point (new-inference path) ---
                # For deltas, the materialize / NaN-masked copy_ / re-quant all happen in
                # `finish_weight_update` (the layer never completes during the chunk updates),
                # so the shadow hooks must be *installed here* and torn down at finish -- a
                # stateful install/teardown, not a `with` around a single call (legacy path).
                # The "delta" WeightTransferEngine owns the shadow; other engines no-op.
                engine = getattr(self, "weight_transfer_engine", None)
                if engine is not None and hasattr(engine, "begin_update"):
                    engine.begin_update()

        self._skyrl_is_checkpoint_format = is_checkpoint_format
        self._skyrl_weight_update_active = True

    def skyrl_update_weights_ipc(self, update_info: dict) -> None:
        """
        Receive and load a single chunk of weights.

        SkyRL packs each chunk's tensors into a single contiguous CUDA buffer and sends
        one IPC handle per rank plus per-param `sizes` metadata. We rebuild
        the packed tensor here, slice it per param, and hand the list to
        model.load_weights (checkpoint format) or copy per-param directly
        (kernel format).

        Args:
            update_info: Dict with keys:
                - names: list[str]
                - dtype_names: list[str]
                - shapes: list[list[int]]
                - sizes: list[int]  (element count per param; used for slicing)
                - ipc_handles_pickled: b64(pickle({gpu_uuid: (func, args)}))
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before update_weights_ipc.")

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. " "Please set weight_transfer_config to enable weight transfer."
            )

        # --- unpack SkyRL packed CUDA IPC format ---
        import base64
        import pickle

        names = update_info["names"]
        shapes = update_info["shapes"]
        sizes = update_info["sizes"]
        pickled = update_info["ipc_handles_pickled"]
        handles = pickle.loads(base64.b64decode(pickled))

        device_index = torch.cuda.current_device()
        physical_gpu_id = str(torch.cuda.get_device_properties(device_index).uuid)
        if physical_gpu_id not in handles:
            raise ValueError(f"IPC handle not found for GPU UUID {physical_gpu_id}. " f"Available: {list(handles)}")
        func, args = handles[physical_gpu_id]
        # Remap device index to the LOCAL current-device.
        list_args = list(args)
        list_args[6] = device_index
        packed_tensor = func(*list_args)

        weights: list[tuple[str, torch.Tensor]] = []
        offset = 0
        for name, shape, size in zip(names, shapes, sizes):
            weights.append((name, packed_tensor[offset : offset + size].view(*shape)))
            offset += size

        # process_weights_after_loading reads get_current_vllm_config() (e.g.
        # flashinfer_cutlass_moe needs the compilation config to build kernels),
        # and vllm only sets that context around init_device / load_model.
        from vllm.config import set_current_vllm_config

        model = self.model_runner.model
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            if self._skyrl_is_checkpoint_format:
                model.load_weights(weights=weights)
            else:
                for name, weight in weights:
                    param = model.get_parameter(name)
                    param.copy_(weight)

        # Ensure consumption of packed_tensor finishes before we return (and
        # before the sender drops its reference on the next barrier).
        torch.accelerator.synchronize()

    def skyrl_update_weights_nccl(self, update_info: dict) -> None:
        """
        Receive a batched weight update via vLLM's NCCL weight transfer engine.

        Alternative to update_weights_ipc for the broadcast (non-IPC) sender:
        the trainer initiates an NCCL broadcast via
        NCCLWeightTransferEngine.trainer_send_weights, and each inference
        worker calls weight_transfer_engine.receive_weights here.

        Routed through this skyrl wrap (rather than vLLM's native
        /update_weights endpoint) so the load is wrapped with
        set_current_vllm_config — process_weights_after_loading on MoE
        models can otherwise instantiate kernels (e.g. FlashInfer CUTLASS)
        whose __init__ reads get_current_vllm_config().

        TODO: remove once the upstream vLLM patch lands (vllm-project/vllm
        weight-sync-fix), then route via the native /update_weights endpoint.
        https://github.com/vllm-project/vllm/pull/42577
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before update_weights_nccl.")

        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. Please set weight_transfer_config to enable weight transfer."
            )

        from vllm.config import set_current_vllm_config

        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)
        model = self.model_runner.model

        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=model.load_weights,
            )

        torch.accelerator.synchronize()

    def skyrl_finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        For checkpoint-format weights, runs layerwise postprocessing
        (quantization repacking, attention weight processing, etc.).
        Must be called after all update_weights_ipc calls are done.
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before finish_weight_update.")

        if self._skyrl_is_checkpoint_format:
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            model = self.model_runner.model
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                finalize_layerwise_reload(model, self.model_config)

                # --- Delta shard-shadow integration point (new-inference path) ---
                # Mirror of the install in `start_weight_update`: tear the hooks down only after
                # `finalize_layerwise_reload` has materialized + re-quantized the deferred (delta)
                # layers, so the persist (D2H) hook fires before the patches lift.
                engine = getattr(self, "weight_transfer_engine", None)
                if engine is not None and hasattr(engine, "end_update"):
                    engine.end_update()

        self._skyrl_weight_update_active = False
        self._skyrl_is_checkpoint_format = True
