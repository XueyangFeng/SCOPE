# From Local Skills to Long-Horizon Web Tasks: Progressive Skill Exploration for LLM Web Agents

本仓库是论文 From Local Skills to Long-Horizon Web Tasks: Progressive Skill Exploration for LLM Web Agents 的官方实现。

使用简单探索策略训练 Web 智能体仍然很有挑战：成功轨迹很少，而稀疏的最终奖励使智能体很难从失败尝试中提取有用反馈。

受技能习得理论启发，我们提出 Skill Composition Progressive Exploration (SCOPE)，这是一个面向长时域 Web 智能体的“由短到长”学习框架。SCOPE 将复杂任务探索分解为一个渐进扩展过程：从局部技能逐步扩展到完整任务解决能力。具体来说，SCOPE 通过三个步骤迭代探索局部技能：技能发现 从历史部署中提取可复用的局部技能和失败经验；技能编排 将发现的技能组合成高级技能链；技能扩展 使用与当前技能链一致的失败经验，引导智能体探索新的子技能。通过这一过程，SCOPE 在多次部署中逐步构建完成任务的技能链，并通过策略内化将其内化到智能体策略中。

Scope 将 benchmark runner 和方法代码分开。本仓库包含 Scope pipeline、prompt、数据集 adapter，以及用于连接上游环境的小型 runner overlay/integration。本仓库不包含完整 benchmark 部署、模型权重、原始数据集或实验日志。

## 1. 环境安装

创建 conda 环境并安装 Scope：

```bash
git clone <scope-repo-url>
cd scope
```

我们强烈建议您为不同的数据集安装不同的环境：

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

## 2. 配置环境变量

创建 `.env` 文件：

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

## 3. 运行 Scope

### Automatic Multi-Round Rollout

运行自动多轮 Scope rollout：

```bash
PYTHONPATH=src python -m scope.<dataset>.run_sequence
```

指定 sequence config：

```bash
PYTHONPATH=src python -m scope.<dataset>.run_sequence --config <sequence-config-path>
```

默认 sequence config：

```text
src/scope/webarena/configs/sequence.yaml
src/scope/webshop/configs/sequence.yaml
src/scope/webvoyager/configs/sequence.yaml
```

### Manual Step-by-Step Rollout Pipeline

Scope 目前支持手动运行单轮 rollout pipeline。每一轮对应 SCOPE 的三个核心步骤和一次环境部署：

```text
Skill Discovery -> Skill Orchestration -> Skill Expansion -> Environment Rollout
```

在代码中，对应的 stage 名称为：

```text
refresh_memory -> select -> update -> rollout
```

运行完整单轮：

```bash
PYTHONPATH=src python -m scope.<dataset>.run_round --stage all
```

`<dataset>` 可选：

```text
webarena
webshop
webvoyager
```

也可以逐阶段运行：

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

指定 config：

```bash
PYTHONPATH=src python -m scope.<dataset>.run_round --config <config-path> --stage all
```

**Parameters**:

```text
--stage      运行阶段：Skill Discovery / Skill Orchestration / Skill Expansion / Environment Rollout
--config     pipeline config 路径；不指定时使用对应数据集的默认 config
```

默认 config：

```text
src/scope/webarena/configs/pipeline.yaml
src/scope/webshop/configs/pipeline.yaml
src/scope/webvoyager/configs/pipeline.yaml
```

### Policy Internalization Data

从多轮 rollout 结果中提取用于 policy internalization 的训练数据：

```bash
PYTHONPATH=src python -m scope.<dataset>.export_train_data
```

指定导出 config：

```bash
PYTHONPATH=src python -m scope.<dataset>.export_train_data --config <export-config-path>
```

默认导出 config：

```text
src/scope/webarena/configs/export_train_data.yaml
src/scope/webshop/configs/export_train_data.yaml
src/scope/webvoyager/configs/export_train_data.yaml
```

默认输出：

```text
artifacts/<dataset>/train_data/scope_rollout_sft.json
```

### Policy Internalization Training

使用 LLaMA-Factory 训练 policy：

```bash
export LLAMA_FACTORY_ROOT=/path/to/LLaMA-Factory
export SCOPE_BASE_MODEL=/path/to/base-or-previous-policy
export SCOPE_TRAIN_DATA_DIR=/path/to/LLaMA-Factory/data
export SCOPE_OUTPUT_DIR=/path/to/output-policy

bash scripts/train_policy.sh training/<dataset>/scope_sft.yaml
```

训练配置：

```text
training/webarena/scope_sft.yaml
training/webshop/scope_sft.yaml
training/webvoyager/scope_sft.yaml
```

## Acknowledgments

THUDM/VisualAgentBench

WebShop 来自hpl和verl-agent

webvoyager训练集来自 webvoyager和openwebvoyager
