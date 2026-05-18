# Scope WebShop HPL Integration

This directory contains a compact WebShop-only runner adapted from HPL for
Scope rollout. It keeps the WebShop evaluation loop and Scope prefix/guided
rollout hooks, and removes unrelated ALFWorld and InterCode-SQL code.

External WebShop data and services are not included. Prepare WebShop from
`verl-agent`, then set:

```bash
cd integrations
git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent/agent_system/environments/env_package/webshop/webshop
./setup.sh -d all

cd ../../../../../..
export WEBSHOP_ROOT=integrations/verl-agent/agent_system/environments/env_package/webshop/webshop
```

Scope invokes this runner through:

```bash
PYTHONPATH=src python -m scope.webshop.run_round --stage rollout
```

The direct runner entrypoint is:

```bash
python evaluation.py \
  --exp_path ./configs/task \
  --exp_config webshop_full \
  --agent_path ./configs/model \
  --agent_config openai
```

Scope-specific intervention logic lives in:

```text
evaluation.py
utils/dcrl_runtime.py
```
