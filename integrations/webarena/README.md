# Scope WebArena Overlay

This directory is an overlay for `THUDM/VisualAgentBench`'s
`VAB-WebArena-Lite` runner. It is not a standalone copy of WebArena or
VisualAgentBench.

Prepare the upstream runner first, then apply the Scope files:

```bash
git clone https://github.com/THUDM/VisualAgentBench.git
cd VisualAgentBench/VAB-WebArena-Lite
# Follow the upstream README to prepare WebArena-Lite.

cd /path/to/scope
bash integrations/webarena/apply_overlay.sh /path/to/VisualAgentBench/VAB-WebArena-Lite
export VAB_WEBARENA_ROOT=/path/to/VisualAgentBench/VAB-WebArena-Lite
export SCOPE_WEBARENA_WRAPPER_CONFIG=src/scope/webarena/configs/webrl_llama_wrapper.yaml
```

The overlay keeps only the files that differ from the upstream runner. It
supports the `webrl_llama` and `webrl_qwen` profiles. The `webrl_llama` profile
uses upstream `agent/prompts/jsons/p_webrl.json`; `webrl_qwen` adds
`agent/prompts/jsons/p_webrl_qwen.json`.

Scope-specific rollout intervention prompts live in
`src/scope/webarena/rollout_prompts/`.
