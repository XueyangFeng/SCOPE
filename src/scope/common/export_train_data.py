from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scope.common.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Scope rollout trajectories for policy internalization.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("sharegpt", "sharegpt_per_step", "jsonl"), default="sharegpt")
    parser.add_argument("--include-failed", action="store_true", default=False)
    parser.add_argument("--all-success", action="store_true", default=False)
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def msg_to_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def to_sharegpt_conversations(messages: list[Any]) -> list[dict[str, str]]:
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    converted: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = role_map.get(str(msg.get("role", "")))
        if role is None:
            continue
        text = msg_to_text(msg)
        if text:
            converted.append({"from": role, "value": text})
    while converted and converted[-1]["from"] != "gpt":
        converted.pop()
    return converted


def to_sharegpt_per_step(messages: list[Any]) -> list[dict[str, Any]]:
    turns = to_sharegpt_conversations(messages)
    return [{"conversations": turns[: idx + 1]} for idx, turn in enumerate(turns) if turn["from"] == "gpt"]


def save_samples(path: Path, samples: list[dict[str, Any]], output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        text = "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples)
        path.write_text((text + "\n") if text else "", encoding="utf-8")
    else:
        path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def success_value(raw: Any) -> bool:
    return raw in {True, 1, 1.0, "1", "true", "True", "SUCCESS", "success"}


def read_webarena_candidate(task_path: Path, run_label: str) -> dict[str, Any] | None:
    try:
        data = load_json(task_path)
    except Exception:
        return None
    messages = data.get("messages") or data.get("conversation") or data.get("conversations")
    if not isinstance(messages, list):
        actions = data.get("actions") or []
        if not isinstance(actions, list):
            actions = []
        messages = [
            {"role": "user", "content": f"WebArena task {task_path.stem}."},
            {"role": "assistant", "content": "\n".join(str(action) for action in actions)},
        ]
    score = data.get("score")
    success = success_value(data.get("success")) if "success" in data else (float(score or 0.0) > 0.5)
    return {"task_id": task_path.stem, "run": run_label, "messages": messages, "success": success, "score": score, "steps": len(messages)}


def read_webshop_candidate(task_path: Path, run_label: str) -> dict[str, Any] | None:
    try:
        data = load_json(task_path)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    terminal = data[-1] if data and isinstance(data[-1], dict) else {}
    reward = float(terminal.get("reward") or terminal.get("score") or 0.0)
    messages = [item for item in data[:-1] if isinstance(item, dict) and item.get("role") in {"user", "assistant", "system"}]
    return {"task_id": task_path.stem, "run": run_label, "messages": messages, "success": reward >= 1.0, "score": reward, "steps": int(terminal.get("steps") or len(messages))}


def read_webvoyager_candidate(task_dir: Path, run_label: str) -> dict[str, Any] | None:
    msg_path = task_dir / "interact_messages.json"
    if not msg_path.exists():
        return None
    try:
        messages = load_json(msg_path)
    except Exception:
        return None
    eval_path = task_dir / "eval_result.json"
    eval_data = load_json(eval_path) if eval_path.exists() else {}
    result = eval_data.get("result") if isinstance(eval_data, dict) else None
    return {
        "task_id": task_dir.name.removeprefix("task"),
        "run": run_label,
        "messages": messages if isinstance(messages, list) else [],
        "success": result == 1,
        "score": result,
        "steps": sum(1 for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant") if isinstance(messages, list) else 0,
    }


def discover_candidates(dataset: str, run_specs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for idx, spec in enumerate(run_specs):
        label = str(spec.get("label") or f"run{idx}")
        path = Path(str(spec.get("path") or ""))
        if not path.exists():
            continue
        candidates: list[dict[str, Any] | None] = []
        if dataset == "webvoyager":
            candidates = [read_webvoyager_candidate(task_dir, label) for task_dir in sorted(path.iterdir()) if task_dir.is_dir() and task_dir.name.startswith("task")]
        elif dataset == "webshop":
            candidates = [read_webshop_candidate(task_path, label) for task_path in sorted(path.glob("*.json"))]
        elif dataset == "webarena":
            candidates = [read_webarena_candidate(task_path, label) for task_path in sorted(path.glob("*.json"))]
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        for candidate in candidates:
            if not candidate:
                continue
            out.setdefault(str(candidate["task_id"]), []).append(candidate)
    return out


def choose_best(candidates: list[dict[str, Any]], include_failed: bool, all_success: bool) -> list[dict[str, Any]]:
    pool = candidates if include_failed else [item for item in candidates if item["success"]]
    if not pool:
        return []
    if all_success:
        return pool
    return [
        max(
            pool,
            key=lambda item: (
                bool(item["success"]),
                float(item["score"] or 0.0) if item["score"] is not None else 0.0,
                -int(item["steps"] or 0),
            ),
        )
    ]


def default_output_path(cfg: dict[str, Any], output_format: str) -> Path:
    path = cfg.get("output")
    if path:
        return Path(str(path))
    dataset = str(cfg.get("dataset") or "scope")
    ext = ".jsonl" if output_format == "jsonl" else ".json"
    return Path("artifacts") / dataset / "train_data" / f"scope_rollout_sft{ext}"


def export_from_config(cfg: dict[str, Any], *, output: Path | None, output_format: str, include_failed: bool, all_success: bool) -> dict[str, Any]:
    dataset = str(cfg.get("dataset") or "")
    run_specs = cfg.get("runs") or []
    if not isinstance(run_specs, list) or not run_specs:
        raise ValueError("Export config requires a non-empty runs list")
    candidates_by_task = discover_candidates(dataset, [dict(item) for item in run_specs if isinstance(item, dict)])
    samples: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    skipped_no_success = 0
    for task_id in sorted(candidates_by_task):
        selected = choose_best(candidates_by_task[task_id], include_failed, all_success)
        if not selected:
            skipped_no_success += 1
            continue
        for item in selected:
            if output_format == "sharegpt_per_step":
                step_samples = to_sharegpt_per_step(item["messages"])
                for sample in step_samples:
                    sample.update({"id": task_id, "source": item["run"], "reward": 1.0 if item["success"] else 0.0})
                samples.extend(step_samples)
            else:
                conversations = to_sharegpt_conversations(item["messages"])
                if not conversations:
                    continue
                samples.append({"id": task_id, "conversations": conversations, "source": item["run"], "reward": 1.0 if item["success"] else 0.0})
            source_counts[item["run"]] = source_counts.get(item["run"], 0) + 1
    out_path = output or default_output_path(cfg, output_format)
    save_samples(out_path, samples, output_format)
    return {
        "dataset": dataset,
        "tasks_seen": len(candidates_by_task),
        "samples": len(samples),
        "skipped_no_success": skipped_no_success,
        "source_counts": source_counts,
        "output": str(out_path),
    }


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    summary = export_from_config(
        cfg,
        output=args.output,
        output_format=args.format,
        include_failed=args.include_failed,
        all_success=args.all_success,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
