# Runner Integrations

This directory contains compact runner integrations used by Scope rollout
adapters. WebArena is provided as an overlay on top of upstream
VisualAgentBench/WebArena-Lite.

It intentionally excludes large benchmark data, source datasets, model weights,
browser state, run logs, cached downloads, and historical experiment outputs.
Prepare each benchmark environment following its upstream license and setup
instructions, then point Scope configs or environment variables to the local
runner/data locations.

Runner roots:

```bash
export VAB_WEBARENA_ROOT=/path/to/VisualAgentBench/VAB-WebArena-Lite
export SCOPE_WEBARENA_WRAPPER_CONFIG=src/scope/webarena/configs/webrl_llama_wrapper.yaml
export HPL_ROOT=integrations/webshop_hpl
export WEBVOYAGER_ROOT=integrations/webvoyager
```
