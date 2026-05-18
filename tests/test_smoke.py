from pathlib import Path

from scope.common.config import build_round_spec, load_config
from scope.common.llm import load_llm_config
from scope.common.render import render_template
from scope.common.stages import run_stage
from scope.common.types import PipelineContext
from scope.common.store import MemoryStoreLayout
from scope.registry import get_adapter


def test_webarena_smoke() -> None:
    config_path = Path("src/scope/webarena/configs/pipeline.yaml")
    config = load_config(config_path)
    adapter = get_adapter("webarena")
    assert config["dataset"] == "webarena"
    assert adapter.name == "webarena"
    summary = adapter.discover_tasks(PipelineContext("webarena", config_path, config))
    assert summary.total == 0


def test_round_spec_and_store_layout() -> None:
    spec = build_round_spec({"iter": 2, "run": 1}, stage="update")
    layout = MemoryStoreLayout(Path("artifacts/demo/memory_store"))
    assert spec.iter_id == 2
    assert spec.run_id == 1
    assert spec.stage == "update"
    assert layout.prefix_memory_path("42").as_posix().endswith("memory_store/42/prefix_memory.json")
    assert layout.memory_update_path("42").as_posix().endswith("memory_store/42/memory_update.jsonl")


def test_webvoyager_missing_eval_is_failed(tmp_path: Path) -> None:
    task_dir = tmp_path / "results" / "taskAmazon--0"
    task_dir.mkdir(parents=True)
    adapter = get_adapter("webvoyager")
    ctx = PipelineContext("webvoyager", Path("config.yaml"), {"results_dir": str(tmp_path / "results")})
    summary = adapter.discover_tasks(ctx)
    assert summary.total == 1
    assert len(summary.failed) == 1


def test_webshop_stage_writes_stub_outputs(tmp_path: Path) -> None:
    trace_dir = tmp_path / "input"
    trace_dir.mkdir()
    (trace_dir / "10001.json").write_text('{"reward": 0.0}', encoding="utf-8")
    cfg = {
        "memory_root": str(tmp_path / "memory_store"),
        "source_trace_dir": str(trace_dir),
        "rollout_output_dir": str(tmp_path / "rollout"),
    }
    adapter = get_adapter("webshop")
    ctx = PipelineContext("webshop", Path("pipeline.yaml"), cfg)
    results = run_stage(ctx, adapter, "all")
    assert [result.stage for result in results] == ["refresh_memory", "select", "update", "rollout"]
    assert (tmp_path / "memory_store" / "10001" / "prefix_memory.json").exists()
    assert (tmp_path / "memory_store" / "10001" / "selected_anchor.jsonl").exists()
    assert (tmp_path / "memory_store" / "10001" / "memory_update.jsonl").exists()
    assert (tmp_path / "rollout" / "10001.json").exists()


def test_render_and_llm_config_defaults() -> None:
    rendered = render_template("Goal={{goal}}", {"goal": "demo"})
    llm = load_llm_config({"llm": {"enabled": False, "model": "demo-model"}})
    assert rendered == "Goal=demo"
    assert llm.enabled is False
    assert llm.model == "demo-model"


def test_webshop_rollout_dry_run_command() -> None:
    config_path = Path("src/scope/webshop/configs/pipeline.yaml")
    config = load_config(config_path)
    adapter = get_adapter("webshop")
    ctx = PipelineContext("webshop", config_path, config)
    result = run_stage(ctx, adapter, "rollout", dry_run=True)[0]
    assert "${HPL_ROOT}" in result.command
    assert "--prefix_root" in result.command
