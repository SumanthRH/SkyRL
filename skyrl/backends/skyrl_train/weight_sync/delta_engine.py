"""vLLM ``WeightTransferEngine`` for sparse bf16 delta sync (new inference path).

This is the receive side of delta weight sync. vLLM owns it: the server is built with
``WeightTransferConfig(backend="delta")`` and this engine is constructed by vLLM's
``WeightTransferEngineFactory`` (registered lazily in
``inference_servers/new_inference_worker_wrap.py``).

Transport is pluggable (see ``delta_transport.py``), selected at init by
``DeltaWeightTransferInitInfo.transport``:

* ``"nccl"`` -- broadcast over vLLM's ``PyNcclCommunicator`` (we subclass
  ``NCCLWeightTransferEngine`` for the group setup); the trainer joins via
  ``NCCLWeightTransferEngine.trainer_init`` and broadcasts the sparse ``(values[, positions])``
  payload directly over the group.
* ``"disk"`` -- the trainer publishes versioned safetensors files to a shared filesystem and
  every worker reads them; no process group is created.

Either way the trainer drives the send itself (see
``DeltaWeightTransferSender._send_chunks_vllm_native``), so we keep
:meth:`trainer_send_weights` a pass-through (raises).

What the engine does each update:

* Owns the per-worker :class:`ShardShadow` (pinned-CPU bf16 master of *this rank's shard*).
* :meth:`begin_update` / :meth:`end_update` install / tear down the shadow's reload hooks
  around the whole ``start_weight_update`` -> ``finish_weight_update`` window (the worker wrap
  calls them, since for deltas the layers only materialize + re-quant in
  ``finalize_layerwise_reload``).
* :meth:`receive_weights` broadcast-receives the sparse payload, rebuilds a fresh **CPU** full
  tensor per touched param (dense for a seed, ``NaN``-everywhere-except-changed for a delta),
  and feeds them to ``model.load_weights`` incrementally. vLLM's per-parameter
  ``weight_loader`` narrows full->shard; the shadow's NaN-masked ``copy_`` overwrites only the
  changed shard positions on top of the staged previous shard, then re-quantizes -- so this
  works for quantized inference too.

v1 scope mirrors the legacy path: bf16-only transfer; every parameter is still yielded each
sync, so only bandwidth (not re-quant compute) is reduced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import WeightTransferUpdateInfo
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
)

from skyrl.backends.skyrl_train.weight_sync.delta_transport import (
    DiskDeltaTransport,
    NcclDeltaTransport,
)
from skyrl.backends.skyrl_train.weight_sync.delta_utils import (
    ShardShadow,
    build_full_delta,
    build_full_seed,
    iter_unpack,
    verify_delta_checksum,
)
from skyrl.train.utils.utils import str_to_torch_dtype


@dataclass
class DeltaWeightTransferInitInfo(NCCLWeightTransferInitInfo):
    """Init info for the delta backend: transport selection + the bf16 transfer dtype.

    The NCCL group params (``master_address``/``master_port``/``world_size``) are inherited
    and only used when ``transport == "nccl"``; the disk transport ignores them.
    """

    # Default required so it can follow the (defaulted) base layout; always supplied in
    # practice via ``DeltaInitInfo.to_api_payload``.
    model_dtype_name: str = "bfloat16"
    transport: str = "nccl"
    """Payload carrier: ``"nccl"`` (broadcast) or ``"disk"`` (shared-FS safetensors)."""
    sync_dir: str = ""
    """Shared-FS directory for ``transport="disk"`` (ignored for NCCL)."""
    keep_files: bool = False
    """Keep each version's delta files after the sync (``transport="disk"`` only)."""


@dataclass
class DeltaWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """One delta update (the transport-agnostic control-plane manifest for a chunk).

    ``counts[i]`` is the number of changed elements for parameter ``i`` (the full element
    count for a seed). The actual positions/values travel out of band via the transport
    (NCCL broadcast or a disk read), not in this (HTTP) request. ``checksum`` is the CRC32
    the trainer computed over the packed payload bytes (verified before apply); ``version``
    and ``chunk_index`` route a disk read to the right file (ignored by NCCL).
    """

    names: list[str] = field(default_factory=list)
    dtype_names: list[str] = field(default_factory=list)
    shapes: list[list[int]] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    is_seed: bool = False
    checksum: int = 0
    version: int = 0
    chunk_index: int = 0

    def __post_init__(self) -> None:
        # The base ``WeightTransferUpdateInfo`` is a plain dataclass without ``__post_init__``;
        # only chain up if a parent actually defines one.
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        n = len(self.names)
        if len(self.dtype_names) != n:
            raise ValueError(f"`dtype_names` must align with `names`: got {len(self.dtype_names)} and {n}")
        if len(self.shapes) != n:
            raise ValueError(f"`shapes` must align with `names`: got {len(self.shapes)} and {n}")
        if len(self.counts) != n:
            raise ValueError(f"`counts` must align with `names`: got {len(self.counts)} and {n}")


class DeltaWeightTransferEngine(NCCLWeightTransferEngine):
    """Apply sparse bf16 deltas to a (possibly quantized) vLLM model via a sharded bf16 shadow."""

    init_info_cls = DeltaWeightTransferInitInfo
    update_info_cls = DeltaWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        parallel_config: ParallelConfig,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, parallel_config, model)
        # Per-worker pinned-CPU bf16 shadow of this rank's local shard.
        self._shadow = ShardShadow()
        self._model_dtype: torch.dtype | None = None
        # Payload carrier (NCCL broadcast or shared-FS disk), built in init_transfer_engine.
        self._transport: NcclDeltaTransport | DiskDeltaTransport | None = None

    # ---- lifecycle -------------------------------------------------------------------

    def init_transfer_engine(self, init_info: DeltaWeightTransferInitInfo) -> None:
        dtype = str_to_torch_dtype(init_info.model_dtype_name)
        if dtype != torch.bfloat16:
            raise ValueError(
                f"Delta weight sync requires a bf16 transfer dtype (the shadow master is "
                f"bf16); got {init_info.model_dtype_name!r}."
            )
        self._model_dtype = dtype

        device = torch.device("cuda", torch.cuda.current_device())
        if init_info.transport == "nccl":
            # Reuse NCCL's StatelessProcessGroup / PyNcclCommunicator setup (rank derived
            # from parallel_config + rank_offset), then wrap the group as a transport.
            super().init_transfer_engine(init_info)
            self._transport = NcclDeltaTransport(self.model_update_group, device)
        elif init_info.transport == "disk":
            # No process group: the trainer publishes files to the shared FS and every worker
            # reads them, so there is nothing to rendezvous on at init.
            self._transport = DiskDeltaTransport(init_info.sync_dir, init_info.keep_files)
        else:
            raise ValueError(f"Unsupported delta transport {init_info.transport!r}; expected 'nccl' or 'disk'.")

        # Prime the bf16 master now. ``initialize`` prefers the pre-quantization bf16 master
        # captured at model load (by the worker wrap's two capture seams -- the per-layer
        # ``_layerwise_process`` wrap for meta-device online quant and the model-level
        # ``process_weights_after_loading`` wrap otherwise), which is the only correct base for
        # a quantized model -- the live params here are already fp8. For a bf16 model (or the
        # unquantized params of a quantized one) it falls back to snapshotting the still-bf16
        # live params, which is equally correct. Either way this must happen now, outside an
        # update: on the new path
        # ``begin_update`` runs after ``initialize_layerwise_reload`` has moved the model to
        # meta, so it could not snapshot there.
        self._shadow.initialize(self.model)

    def begin_update(self) -> None:
        """Install the shard-shadow reload hooks for the upcoming update.

        Called by the worker wrap in ``start_weight_update`` (inside the
        ``set_current_vllm_config`` context), after ``initialize_layerwise_reload``.
        """
        self._shadow.install(self.model)

    def end_update(self) -> None:
        """Tear down the shard-shadow hooks after ``finalize_layerwise_reload`` re-quantizes."""
        self._shadow.uninstall()

    def shutdown(self) -> None:
        self._shadow.shutdown()
        self._shadow.clear()
        if self._transport is not None:
            self._transport.teardown()
            self._transport = None
        super().shutdown()

    # ---- receive ---------------------------------------------------------------------

    def receive_weights(
        self,
        update_info: DeltaWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Obtain the sparse payload (via the transport) and feed full (NaN-narrow) tensors.

        ``load_weights`` is ``model.load_weights`` (checkpoint path). We build a fresh **CPU**
        full tensor per touched param -- a separate object each time, since the layerwise
        reload buffers tensors until their layer is processed in ``finalize_layerwise_reload``.
        We force ``device=cpu`` because the worker wrap runs ``receive_weights`` inside a
        ``torch.device("cuda")`` context (so an unqualified allocation would land the full
        transient on GPU). The payload carrier (NCCL broadcast vs disk read) is fully owned by
        ``self._transport``; this method is transport-agnostic.
        """
        if self._transport is None:
            raise RuntimeError("Delta weight transfer not initialized. Call init_transfer_engine() first.")

        cpu = torch.device("cpu")
        counts = update_info.counts
        total = int(sum(counts))

        values_cpu, positions_cpu = self._transport.receive(
            total=total,
            is_seed=update_info.is_seed,
            version=update_info.version,
            chunk_index=update_info.chunk_index,
        )
        # Integrity guard: the trainer's CRC32 must match the bytes we obtained, regardless of
        # whether they came over NCCL or off the shared filesystem.
        verify_delta_checksum(update_info.checksum, positions_cpu, values_cpu)

        if update_info.is_seed:
            for i, offset, count in iter_unpack(counts):
                shape = update_info.shapes[i]
                values = values_cpu[offset : offset + count]
                load_weights([(update_info.names[i], build_full_seed(shape, values, device=cpu))])
            return

        assert positions_cpu is not None
        for i, offset, count in iter_unpack(counts):
            shape = update_info.shapes[i]
            positions = positions_cpu[offset : offset + count]
            values = values_cpu[offset : offset + count]
            load_weights([(update_info.names[i], build_full_delta(shape, positions, values, device=cpu))])

    # ---- trainer side ----------------------------------------------------------------

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        """Pass-through: the SkyRL trainer broadcasts the sparse payload itself.

        See ``DeltaWeightTransferSender._send_chunks_vllm_native``, which diffs against its
        snapshot and broadcasts ``values`` (and ``positions`` for a delta) directly over the
        NCCL group created via ``NCCLWeightTransferEngine.trainer_init``.
        """
        raise NotImplementedError(
            "DeltaWeightTransferEngine.trainer_send_weights is intentionally a pass-through; "
            "the SkyRL trainer broadcasts the sparse (values[, positions]) payload directly "
            "over the NCCL group (see DeltaWeightTransferSender._send_chunks_vllm_native)."
        )
