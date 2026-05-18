# WebArena

This directory contains the WebArena/WebArena-Lite integration for Scope.

## Setup

Install and configure a WebArena-compatible backend through
VisualAgentBench/WebArena-Lite. Scope provides an overlay at
`integrations/webarena`; apply it to the upstream runner before rollout.

```bash
git clone https://github.com/THUDM/VisualAgentBench.git
# Follow VisualAgentBench/VAB-WebArena-Lite setup instructions.
bash integrations/webarena/apply_overlay.sh /path/to/VisualAgentBench/VAB-WebArena-Lite
export VAB_WEBARENA_ROOT=/path/to/VisualAgentBench/VAB-WebArena-Lite
export SCOPE_WEBARENA_WRAPPER_CONFIG=src/scope/webarena/configs/webrl_llama_wrapper.yaml
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

The rollout builder expects a VisualAgentBench-style script such as
`wa_parallel_run_yi.sh`. The overlay contains only Scope changes relative to
the upstream runner, not the full WebArena service deployment or historical
experiment outputs.

The base rollout prompt is provided by the original VisualAgentBench runner.
Scope only includes the failure-guided intervention prompt used by
`guided_think_mode=auto`.

## Pipeline

```bash
PYTHONPATH=src python -m scope.webarena.run_round --stage refresh_memory --dry-run
PYTHONPATH=src python -m scope.webarena.run_round --stage select --dry-run
PYTHONPATH=src python -m scope.webarena.run_round --stage update --dry-run
PYTHONPATH=src python -m scope.webarena.run_round --stage rollout --dry-run
PYTHONPATH=src python -m scope.webarena.run_round --stage all --dry-run
```

Drop `--dry-run` after the WebArena services, browser backend, and model
endpoint are configured.

Default config: `configs/pipeline.yaml`

Wrapper configs:

```text
configs/webrl_llama_wrapper.yaml
configs/webrl_qwen_wrapper.yaml
```

Use `SCOPE_WEBARENA_WRAPPER_CONFIG` to switch between them.

## Files

- `adapter.py`: discovers WebArena tasks from `actions/*.json` or `traces/*.jsonl`.
- `generator.py`: renders the WebArena memory-generation prompt and calls the LLM.
- `selector.py`: selects the reusable prefix anchor for the next rollout.
- `update.py`: updates selected memory from the newly observed suffix trajectory.
- `prompts/`: generator, update, and selector prompts.
- `rollout_prompts/`: Scope failure-guided intervention prompt.
- `rollout.py`: builds the WebArena rollout command.
- `run_round.py`: dataset-local CLI entrypoint.
