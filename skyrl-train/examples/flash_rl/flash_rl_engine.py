import os
from typing import List, Any, Dict, Optional
import ray
import torch
import vllm
from skyrl_train.inference_engines.vllm.vllm_engine import WorkerWrap, VLLMInferenceEngine
from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine
from packaging import version
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
)


class FlashRLWorkerWrap(WorkerWrap):

    def patch_vllm_load_weights(self):

        from loguru import logger
        from examples.flash_rl.flash_rl.vllm_patch import bond_method_to_cls, load_flashrl_config
        from examples.flash_rl.flash_rl.flash_quantization import get_quantize_fn
        import gc

        config_data = load_flashrl_config(os.environ.get("FLASHRL_CONFIG", None))["configs"][0]

        if "module_attribute_to_preserve" in config_data and not getattr(
            self, "flash_rl_module_attribute_to_preserve", None
        ):
            logger.debug(f"flash_rl module_attribute_to_preserve: {config_data['module_attribute_to_preserve']}")
            self.flash_rl_module_attribute_to_preserve = config_data.get("module_attribute_to_preserve")
        else:
            self.flash_rl_module_attribute_to_preserve = []

        model = self.model_runner.model
        if (not hasattr(model, "beforeflashrl_load_weights")) and (config_data.get("fn", "int8") != "bf16"):
            quant_fn = config_data.get("fn", "int8")
            logger.debug(f"flash_rl quantization function: {quant_fn}")
            flash_quantize_fn = get_quantize_fn(quant_fn)

            # Store the original load_weights function
            original_load_weights = model.load_weights
            model.beforeflashrl_load_weights = original_load_weights

            def hacked_load_weights(
                weights,
            ):
                # print("flash_rl quant load_weights is called")
                setattr(model, "hacked_not_need_process_weights_after_loading", False)

                if len(self.flash_rl_module_attribute_to_preserve) > 0:
                    for _, module in model.named_modules():
                        for attr in self.flash_rl_module_attribute_to_preserve:
                            if torch.is_tensor(getattr(module, attr, None)):
                                # print(f"flash_rl reserving {attr} in module {module}")
                                setattr(module, f"hacked_{attr}", getattr(module, attr))

                existing_params = dict(model.named_parameters())

                hacked_data_dict = {}
                for name, p in existing_params.items():
                    hacked_data_dict[name] = p.data

                assert hasattr(model, "hacked_original_weights_rebuild_keys")

                for name, (shape, stride, dtype, nbytes) in model.hacked_original_weights_rebuild_keys.items():
                    if name in existing_params:
                        existing_params[name].data = torch.empty(shape, dtype=dtype)

                for k, loader_k in model.hacked_recorded_loader.items():
                    for n, loader in loader_k.items():
                        if not hasattr(existing_params[n], k):
                            setattr(existing_params[n], k, bond_method_to_cls(loader, existing_params[n]))

                updated_params = original_load_weights(flash_quantize_fn(weights, None))  # no profile supported for now

                if hasattr(model, "hacked_model_config") and hasattr(model, "hacked_target_device"):
                    from vllm.model_executor.model_loader import utils

                    utils.process_weights_after_loading(model, None, None)
                    setattr(model, "hacked_not_need_process_weights_after_loading", True)
                else:
                    setattr(model, "hacked_not_need_process_weights_after_loading", False)

                skipped_params = list()
                for name, p in model.named_parameters():
                    assert name in hacked_data_dict, f"param {name} is not in hacked_data dict"
                    assert (
                        hacked_data_dict[name].dtype == p.data.dtype
                    ), f"param {name} dtype mismatch: {hacked_data_dict[name].dtype} vs {p.data.dtype}"
                    assert (
                        hacked_data_dict[name].numel() == p.data.numel()
                    ), f"param {name} numel() mismatch: {hacked_data_dict[name].numel()} vs {p.data.numel()}"

                    if name in updated_params:
                        trided_data = torch.as_strided(
                            p.data, hacked_data_dict[name].shape, hacked_data_dict[name].stride()
                        )
                        hacked_data_dict[name].copy_(trided_data)
                    else:
                        skipped_params.append(name)

                    tmp_data = p.data
                    p.data = hacked_data_dict[name]
                    del tmp_data

                if quant_fn not in ["fp8", "fp8_vllm"]:
                    logger.debug(
                        f"flash_rl load_weights skipped params (not accurate for `fp8-vllm`): {skipped_params}"
                    )
                del skipped_params
                del hacked_data_dict
                del existing_params
                gc.collect()
                torch.cuda.empty_cache()

                if len(self.flash_rl_module_attribute_to_preserve) > 0:
                    for _, module in model.named_modules():
                        for attr in self.flash_rl_module_attribute_to_preserve:
                            if torch.is_tensor(getattr(module, attr, None)):
                                assert hasattr(
                                    module, f"hacked_{attr}"
                                ), f"module {module} does not have attribute hacked_{attr}"
                                setattr(module, attr, getattr(module, f"hacked_{attr}"))
                                delattr(module, f"hacked_{attr}")

                return updated_params

            model.load_weights = hacked_load_weights
            logger.debug("Successfully patched the load_weights function of vllm")
        else:
            logger.debug("vllm load_weights patching skipped")


class FlashRLVLLMInferenceEngine(VLLMInferenceEngine):

    def _create_engine(self, *args, **kwargs):
        llm = vllm.LLM(*args, **kwargs)
        if os.environ.get("FLASHRL_CONFIG", None):
            engine = llm.engine if hasattr(llm, "engine") else llm
            engine.collective_rpc("patch_vllm_load_weights")
        return llm


VLLMRayActor = ray.remote(FlashRLVLLMInferenceEngine)


def create_ray_wrapped_inference_engines_flashrl(
    num_inference_engines: int,
    tensor_parallel_size: int,
    model_dtype: str,
    pretrain: str,
    seed: int,
    vllm_v1_disable_multiproc: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    async_engine=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    sampling_params: Optional[Dict[str, Any]] = None,
    tokenizer=None,
    backend="vllm",
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface instances.
    """
    from skyrl_train.utils import ray_noset_visible_devices, get_all_env_variables, get_ray_pg_ready_with_timeout

    assert not async_engine, "`async_engine` is not supported for FlashRL"

    if backend == "vllm":
        import vllm

        assert version.parse(vllm.__version__) >= version.parse("0.8.3"), "SkyRL-Train only supports vLLM >= 0.8.3"
    else:
        raise ValueError(f"Unsupported FlashRL backend: {backend}")

    inference_engine_actors = []
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    # NOTE: we use the ray backend for tensor parallel size > 1 to explicitly manage resource allocation
    # TODO: we should be able to support mp backend by allocating resources at engine level
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all inference engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_inference_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(shared_pg, timeout=30)

    for i in range(num_inference_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i * tensor_parallel_size,
        )

        if backend == "vllm":

            engine = VLLMRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_extension_cls="examples.flash_rl.flash_rl_engine.FlashRLWorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype=model_dtype,
                trust_remote_code=True,
                vllm_v1_disable_multiproc=vllm_v1_disable_multiproc,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=inference_engine_enable_sleep,
                noset_visible_devices=noset_visible_devices,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
                # only need the logprobs for the chosen token if any
                max_logprobs=1,
            )

        inference_engine_actors.append(engine)

    engines = [RayWrappedInferenceEngine(actor_handle) for actor_handle in inference_engine_actors]

    if inference_engine_enable_sleep:
        sleep_refs = [engine.inference_engine_actor.sleep.remote() for engine in engines]
        ray.get(sleep_refs)

    return engines
