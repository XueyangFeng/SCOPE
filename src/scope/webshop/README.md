# WebShop

This directory contains the WebShop/HPL integration for Scope.

## Setup

Install WebShop from `verl-agent`. A minimal adapted HPL runner is bundled at
`integrations/webshop_hpl`.

```bash
cd integrations
git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent/agent_system/environments/env_package/webshop/webshop
./setup.sh -d all

cd ../../../../../..
export HPL_ROOT=integrations/webshop_hpl
export WEBSHOP_ROOT=integrations/verl-agent/agent_system/environments/env_package/webshop/webshop
export DCRL_MODEL=...
export DCRL_BASE_URL=http://localhost:8000/v1
export DCRL_API_KEY=...
```

The bundled integration contains core runner code and configs only. It does not
include large WebShop data, model weights, service state, or past outputs.

The base rollout prompt is provided by the original HPL/WebShop runner. Scope
only includes the failure-guided intervention prompt used by
`guided_think_mode`.

## Pipeline

```bash
PYTHONPATH=src python -m scope.webshop.run_round --stage refresh_memory --dry-run
PYTHONPATH=src python -m scope.webshop.run_round --stage select --dry-run
PYTHONPATH=src python -m scope.webshop.run_round --stage update --dry-run
PYTHONPATH=src python -m scope.webshop.run_round --stage rollout --dry-run
PYTHONPATH=src python -m scope.webshop.run_round --stage all --dry-run
```

Drop `--dry-run` after WebShop data/services and the policy model endpoint are
configured.

Default config: `configs/pipeline.yaml`

## Files

- `adapter.py`: discovers WebShop tasks from JSON traces and parses reward/score.
- `generator.py`: renders the WebShop memory-generation prompt and calls the LLM.
- `selector.py`: selects the reusable prefix anchor for the next rollout.
- `update.py`: updates selected memory from the newly observed suffix trajectory.
- `prompts/`: generator, update, and selector prompts.
- `rollout_prompts/`: Scope failure-guided intervention prompt.
- `rollout.py`: builds the HPL/WebShop `evaluation.py` command.
- `run_round.py`: dataset-local CLI entrypoint.
