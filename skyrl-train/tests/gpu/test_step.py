
import ray
import pytest
import hydra
from omegaconf import DictConfig, OmegaConf

from tests.gpu.utils import init_worker_with_type, make_dummy_experience, validate_cfg
from skyrl_train.utils.utils import print_mem
from skyrl_train.entrypoints.main_base import config_dir


MODEL_NAME = "unsloth/gpt-oss-20b-BF16"
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def get_test_actor_config() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 8

    algorithm_config = OmegaConf.create(cfg.trainer.algorithm)
    algorithm_config.max_seq_len = cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length
    cfg.trainer.algorithm = algorithm_config

    return cfg



cfg = get_test_actor_config()

actor_group = init_worker_with_type(
    "policy",
    shared_pg=None,
    colocate_all=False,
    num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
    cfg=cfg,
)

dummy_experience = make_dummy_experience()
global_step, local_step, accumulation_steps = 0, 0, 1

results = ray.get(
    actor_group.async_run_ray_method(
        "pass_through", "training_step", dummy_experience, global_step, local_step, accumulation_steps
    )
)

memory = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))
memory = memory[0]
print_mem("memory after training step", memory)

for result in results:
    assert isinstance(result, dict), "Result should be a dictionary of training stats"
    assert "policy_loss" in result
    assert "policy_lr" in result
    assert "ppo_clip_ratio" in result
    assert "policy_entropy" in result
    for k, v in result.items():
        assert isinstance(v, (int, float)), f"{k} should be an int or float"