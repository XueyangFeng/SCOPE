# Scope WebVoyager Overlay

This directory contains the Scope overlay for WebVoyager. Clone the upstream
WebVoyager repository, install its dependencies, then apply this overlay.

```bash
cd integrations
git clone https://github.com/MinorJerry/WebVoyager.git
cd WebVoyager
pip install -r requirements.txt

cd ../..
bash integrations/webvoyager_overlay/apply_overlay.sh integrations/WebVoyager

export WEBVOYAGER_ROOT=integrations/WebVoyager
export WEBVOYAGER_TASK_FILE=integrations/WebVoyager/data/WebVoyager_trainset.jsonl
```

Overlay files:

```text
run.py
prompts.py
utils_webarena.py
evaluation/auto_eval.py
```
