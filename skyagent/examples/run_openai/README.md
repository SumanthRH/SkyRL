# SkyAgent + OpenAI

You can run evaluation with any OpenAI-compatible server.


## Running evaluation with Sandbox Fusion

You can spin up a vllm server with: 

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
```


Make sure to set up sandbox fusion following instructions here: https://github.com/bytedance/SandboxFusion. Finally, run evaluation with:

```bash
uv run --isolated --directory . --env-file .env --frozen python ./tests/react_task_tests/test.py --yaml tests/react_task_tests/react_interpreter.yaml --dataset NovaSky-AI/AIME-Repeated-8x-240 --split test
```

### Switching to a different model

At the moment, you need to change the model name in the yaml file and in the run script to grab the right tokenizer. We are actively working on making this experience better.