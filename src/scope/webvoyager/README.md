# WebVoyager

This directory contains the WebVoyager integration for Scope.

## Setup

Install WebVoyager and apply the Scope overlay.

```bash
cd integrations
git clone https://github.com/MinorJerry/WebVoyager.git
cd WebVoyager
pip install -r requirements.txt

cd ../..
bash integrations/webvoyager_overlay/apply_overlay.sh integrations/WebVoyager

export WEBVOYAGER_ROOT=integrations/WebVoyager
export WEBVOYAGER_TASK_FILE=integrations/WebVoyager/data/WebVoyager_trainset.jsonl
export DCRL_MODEL=...
export DCRL_BASE_URL=http://localhost:8000/v1
export DCRL_API_KEY=...
```

The overlay contains Scope runner changes only. It does not include raw task
data, large outputs, cached browser state, or model weights.

The base rollout prompt is defined inside the original WebVoyager runner
(`prompts.py`). Scope only includes the failure-guided intervention prompt.

## Pipeline

```bash
PYTHONPATH=src python -m scope.webvoyager.run_round --stage refresh_memory --dry-run
PYTHONPATH=src python -m scope.webvoyager.run_round --stage select --dry-run
PYTHONPATH=src python -m scope.webvoyager.run_round --stage update --dry-run
PYTHONPATH=src python -m scope.webvoyager.run_round --stage rollout --dry-run
PYTHONPATH=src python -m scope.webvoyager.run_round --stage all --dry-run
```

Drop `--dry-run` after browser dependencies and model endpoints are configured.

Default config: `configs/pipeline.yaml`

## Files

- `adapter.py`: discovers WebVoyager tasks from `results/task*/eval_result.json`.
- `generator.py`: renders the WebVoyager memory-generation prompt and calls the LLM.
- `selector.py`: selects the reusable prefix anchor for the next rollout.
- `update.py`: updates selected memory from the newly observed suffix trajectory.
- `prompts/`: generator, update, and selector prompts.
- `rollout_prompts/`: Scope failure-guided intervention prompt.
- `rollout.py`: builds the WebVoyager `run.py` command.
- `run_round.py`: dataset-local CLI entrypoint.
