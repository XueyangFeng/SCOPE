# From Local Skills to Long-Horizon Web Tasks: Progressive Skill Exploration for LLM Web Agents

This repository is the official implementation of the paper From Local Skills to Long-Horizon Web Tasks: Progressive Skill Exploration for LLM Web Agents.

Training web agents with simple exploration strategies remains challenging: successful trajectories are rare, and sparse final rewards make it difficult for agents to extract useful feedback from failed attempts.

Inspired by skill acquisition theory, we propose Skill Composition Progressive Exploration (SCOPE), a short-to-long learning framework for long-horizon web agents. SCOPE decomposes complex task exploration into a progressive expansion process, growing from local skills toward complete task-solving ability. Specifically, SCOPE iteratively explores local skills through three steps: Skill Discovery extracts reusable local skills and failed experiences from historical deployments; Skill Orchestration composes discovered skills into higher-level skill chains; Skill Expansion uses failed experiences aligned with the current skill chain to guide the agent toward new sub-skills. Through this process, SCOPE gradually builds task-solving skill chains across multiple deployments and internalizes them into the agent policy through policy internalization.

Scope separates benchmark runners from method code. This repository contains the Scope pipeline, prompts, dataset adapters, and lightweight runner overlays/integrations for connecting to upstream environments. It does not include full benchmark deployments, model weights, raw datasets, or experiment logs.

## 1. Environment Setup

Create a conda environment and install Scope:

```bash
git clone <scope-repo-url>
cd scope
```

We strongly recommend using a separate environment for each dataset.

webarena

```bash
conda create -n scope-webarena python=3.10 -y
conda activate scope-webarena

cd integrations
git clone https://github.com/THUDM/VisualAgentBench.git
cd VisualAgentBench/VAB-WebArena-Lite

cd ../../..
bash integrations/webarena/apply_overlay.sh integrations/VisualAgentBench/VAB-WebArena-Lite
```

Expected structure:

```text
scope/
  integrations/
    webarena/
      apply_overlay.sh
      OVERLAY_FILES.txt
      ...
    VisualAgentBench/
      VAB-WebArena-Lite/
        wa_parallel_run_yi.sh
        run_with_prefix.py
        rollout_runtime/
        agent/
        browser_env/
        evaluation_harness/
        llms/
  src/
    scope/
      webarena/
        configs/
          webrl_llama_wrapper.yaml
          webrl_qwen_wrapper.yaml
        rollout_prompts/
          guided_prompt.txt
```

webshop

```bash
conda create -n scope-webshop python=3.10 -y
conda activate scope-webshop

cd integrations
git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent/agent_system/environments/env_package/webshop/webshop
./setup.sh -d all

cd ../../../../../..
cd integrations/webshop_hpl
pip install -r requirements.txt
cd ../..
```

webvoyager

```bash
conda create -n scope-webvoyager python=3.10 -y
conda activate scope-webvoyager

cd integrations
git clone https://github.com/MinorJerry/WebVoyager.git
cd WebVoyager
pip install -r requirements.txt

cd ../..
bash integrations/webvoyager_overlay/apply_overlay.sh integrations/WebVoyager
```

Expected structure:

```text
scope/
  integrations/
    webvoyager_overlay/
      apply_overlay.sh
      OVERLAY_FILES.txt
      overlay/
        run.py
        prompts.py
        utils_webarena.py
        evaluation/
          auto_eval.py
    WebVoyager/
      run.py
      prompts.py
      utils.py
      utils_webarena.py
      evaluation/
        auto_eval.py
      data/
        WebVoyager_trainset.jsonl
  src/
    scope/
      webvoyager/
        configs/
          pipeline.yaml
        rollout_prompts/
          guided_prompt.txt
```

## 2. Configure Environment Variables

Create a `.env` file:

```bash
# common
DCRL_MODEL=<policy-model-name>
DCRL_BASE_URL=<openai-compatible-api-base>
DCRL_API_KEY=<api-key>

# webarena
VAB_WEBARENA_ROOT=integrations/VisualAgentBench/VAB-WebArena-Lite
SCOPE_WEBARENA_WRAPPER_CONFIG=src/scope/webarena/configs/webrl_llama_wrapper.yaml
DCRL_MODEL_PATH=<policy-model-name-or-endpoint-model-id>

# webshop
HPL_ROOT=integrations/webshop_hpl
WEBSHOP_ROOT=integrations/verl-agent/agent_system/environments/env_package/webshop/webshop

# webvoyager
WEBVOYAGER_ROOT=integrations/WebVoyager
WEBVOYAGER_TASK_FILE=integrations/WebVoyager/data/WebVoyager_trainset.jsonl
```

## 3. Running Scope

### Automatic Multi-Round Rollout

Run automatic multi-round Scope rollout:

```bash
PYTHONPATH=src python -m scope.<dataset>.run_sequence
```

Specify a sequence config:

```bash
PYTHONPATH=src python -m scope.<dataset>.run_sequence --config <sequence-config-path>
```

Default sequence configs:

```text
src/scope/webarena/configs/sequence.yaml
src/scope/webshop/configs/sequence.yaml
src/scope/webvoyager/configs/sequence.yaml
```

### Manual Step-by-Step Rollout Pipeline

Scope currently supports manually running a single-round rollout pipeline. Each round corresponds to the three core SCOPE steps and one environment deployment:

```text
Skill Discovery -> Skill Orchestration -> Skill Expansion -> Environment Rollout
```

The corresponding stage names in code are:

```text
refresh_memory -> select -> update -> rollout
```

Run a full single round:

```bash
PYTHONPATH=src python -m scope.<dataset>.run_round --stage all
```

`<dataset>` can be:

```text
webarena
webshop
webvoyager
```

You can also run each stage separately:

```bash
# Skill Discovery
PYTHONPATH=src python -m scope.<dataset>.run_round --stage refresh_memory

# Skill Orchestration
PYTHONPATH=src python -m scope.<dataset>.run_round --stage select

# Skill Expansion
PYTHONPATH=src python -m scope.<dataset>.run_round --stage update

# Environment Rollout
PYTHONPATH=src python -m scope.<dataset>.run_round --stage rollout
```

Specify a config:

```bash
PYTHONPATH=src python -m scope.<dataset>.run_round --config <config-path> --stage all
```

**Parameters**:

```text
--stage      Running stage: Skill Discovery / Skill Orchestration / Skill Expansion / Environment Rollout
--config     Pipeline config path; if omitted, the default config for the selected dataset is used
```

Default configs:

```text
src/scope/webarena/configs/pipeline.yaml
src/scope/webshop/configs/pipeline.yaml
src/scope/webvoyager/configs/pipeline.yaml
```

### Policy Internalization Data

Extract training data for policy internalization from multi-round rollout results:

```bash
PYTHONPATH=src python -m scope.<dataset>.export_train_data
```

Specify an export config:

```bash
PYTHONPATH=src python -m scope.<dataset>.export_train_data --config <export-config-path>
```

Default export configs:

```text
src/scope/webarena/configs/export_train_data.yaml
src/scope/webshop/configs/export_train_data.yaml
src/scope/webvoyager/configs/export_train_data.yaml
```

Default output:

```text
artifacts/<dataset>/train_data/scope_rollout_sft.json
```

### Policy Internalization Training

Train the policy with LLaMA-Factory:

```bash
export LLAMA_FACTORY_ROOT=/path/to/LLaMA-Factory
export SCOPE_BASE_MODEL=/path/to/base-or-previous-policy
export SCOPE_TRAIN_DATA_DIR=/path/to/LLaMA-Factory/data
export SCOPE_OUTPUT_DIR=/path/to/output-policy

bash scripts/train_policy.sh training/<dataset>/scope_sft.yaml
```

Training configs:

```text
training/webarena/scope_sft.yaml
training/webshop/scope_sft.yaml
training/webvoyager/scope_sft.yaml
```

## Acknowledgments

THUDM/VisualAgentBench

WebShop is based on hpl and verl-agent.

The WebVoyager training set is based on WebVoyager and OpenWebVoyager.
