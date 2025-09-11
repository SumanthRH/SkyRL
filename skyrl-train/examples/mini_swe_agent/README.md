## Guide: Mini-SWE-Agent + SkyRL

This directory contains an integration to train a SWE-Agent on the SWE-Bench task using [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent) and SkyRL.

To start training, follow three simple steps:
1) Prepare the SWE-Gym dataset.
2) Configure your environment backend (Podman).
3) Launch training!

Start by following the SkyRL [installation instructions](https://skyrl.readthedocs.io/en/latest/getting-started/installation.html), then enter the `skyrl-train` directory:
```bash
cd SkyRL/skyrl-train
```

## How it works

The Mini-SWE-Agent integration implements a custom `MiniSweAgentGenerator` that uses Mini-SWE-Agent to generate trajectories for SWE-Bench instances. The workflow consists of:

1. **Generation**: Initialize a sandbox environment and generate a trajectory using Mini-SWE-Agent configured with SkyRL's HTTP endpoint, producing a git patch.
2. **Evaluation**: Apply the generated patch to a fresh environment and run the evaluation script to determine if the instance was resolved.

This process runs as Ray tasks to scale generation across cluster nodes.

### 1) Prepare the dataset

We use [SWE-Gym](https://huggingface.co/SWE-Gym), specifically the subset from [SumanthRH/SWE-Gym-Subset](https://huggingface.co/datasets/SumanthRH/SWE-Gym-Subset).

Execute the following command:
```bash
uv run --isolated examples/mini_swe_agent/preprocess_swegym.py --output_dir ~/data/swe_gym_subset
```

### 2) Configure environment backend

**Prerequisites**: Install the required environment backend. By default, we use [Podman](https://podman.io/docs). This can be modified in `examples/mini_swe_agent/swebench.yaml`.

### 3) Launch training

We provide example scripts for different model sizes:

**Qwen3-8B** (requires 1x 8xH100 node):
```bash
bash examples/mini_swe_agent/run_mini_swe_8B.sh
```

**Qwen3-Coder-30B** (requires 2x 8xH100 nodes):
```bash
bash examples/mini_swe_agent/run_mini_swe_30B.sh
```

All training parameters can be modified in the run scripts, such as model choice, GRPO group size, or training batch size.

## Troubleshooting

For issues with SkyRL or the Mini-SWE-Agent integration, please [open an Issue](https://github.com/NovaSky-AI/SkyRL/issues/new).

### Common Issues

- **Context length errors**: If you see `ValueError: The decoder prompt (length xxxx) is longer than the maximum model length`, increase `max_input_length` and `max_generate_length` or reduce steps in `swebench.yaml`.

- **All zero rewards**: If rewards are consistently zero, the task may be too difficult. Consider:
  - Filtering data for a better mix of easy/hard samples
  - Using a stronger base model
  - Increasing `step_limit` in `swebench.yaml`

- **Argument list too long**: For very large git patches exceeding ~1MB, the system hits `ARG_MAX` limits. These patches are assumed to be incorrect.

- **Podman UID errors**: If running in containers with insufficient UIDs, either:
  1. Edit `/etc/subuid` and `/etc/subgid` to use range `100000-1100000`
  2. Set `ignore_chown_errors=true` in Podman's containers.conf

## Configuration

The main configuration file is `examples/mini_swe_agent/swebench.yaml`, which controls:
- Environment backend settings
- Step limits for agent execution
- Tool configurations for Mini-SWE-Agent

See the [full documentation](../../docs/examples/mini_swe_agent.rst) for detailed implementation details.