# Policy Internalization

Scope uses LLaMA-Factory for SFT-style policy internalization.

1. Export rollout training data:

```bash
PYTHONPATH=src python -m scope.<dataset>.export_train_data
```

2. Register the exported dataset in `LLaMA-Factory/data/dataset_info.json`.

3. Run training:

```bash
export LLAMA_FACTORY_ROOT=/path/to/LLaMA-Factory
export SCOPE_BASE_MODEL=/path/to/base-or-previous-policy
export SCOPE_TRAIN_DATA_DIR=/path/to/LLaMA-Factory/data
export SCOPE_OUTPUT_DIR=/path/to/output-policy

bash scripts/train_policy.sh training/<dataset>/scope_sft.yaml
```
