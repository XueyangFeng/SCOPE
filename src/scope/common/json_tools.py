from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


def strip_code_fences(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    return raw


def extract_balanced_json_candidate(text: str) -> str:
    start = -1
    opener = ""
    for idx, ch in enumerate(text):
        if ch in "{[":
            start = idx
            opener = ch
            break
    if start < 0:
        raise ValueError("No JSON object found in model output")

    closer_for = {"{": "}", "[": "]"}
    stack = [closer_for[opener]]
    in_string = False
    escape = False

    for idx in range(start + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(closer_for[ch])
            continue
        if ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()
                if not stack:
                    return text[start : idx + 1]
            continue

    return text[start:] + "".join(reversed(stack))


def _consume_string(text: str, start: int) -> tuple[str, int]:
    idx = start + 1
    escape = False
    while idx < len(text):
        ch = text[idx]
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            return text[start : idx + 1], idx + 1
        idx += 1
    return text[start:] + '"', len(text)


def _consume_number(text: str, start: int) -> tuple[str, int]:
    idx = start
    while idx < len(text) and text[idx] in "-+0123456789.eE":
        idx += 1
    return text[start:idx], idx


def _consume_literal(text: str, start: int) -> tuple[str, int]:
    idx = start
    while idx < len(text) and text[idx].isalpha():
        idx += 1
    return text[start:idx], idx


def _insert_missing_commas(text: str) -> str:
    out: list[str] = []
    idx = 0
    last_token_kind = "start"
    value_end_kinds = {"string", "number", "literal", "close"}
    value_start_kinds = {"string", "number", "literal", "open"}

    while idx < len(text):
        ch = text[idx]
        if ch.isspace():
            out.append(ch)
            idx += 1
            continue

        if ch == '"':
            token, next_idx = _consume_string(text, idx)
            token_kind = "string"
        elif ch in "{[":
            token, next_idx = ch, idx + 1
            token_kind = "open"
        elif ch in "}]":
            token, next_idx = ch, idx + 1
            token_kind = "close"
        elif ch == ":":
            token, next_idx = ch, idx + 1
            token_kind = "colon"
        elif ch == ",":
            token, next_idx = ch, idx + 1
            token_kind = "comma"
        elif ch in "-0123456789":
            token, next_idx = _consume_number(text, idx)
            token_kind = "number"
        elif ch.isalpha():
            token, next_idx = _consume_literal(text, idx)
            token_kind = "literal"
        else:
            token, next_idx = ch, idx + 1
            token_kind = "other"

        if token_kind in value_start_kinds and last_token_kind in value_end_kinds:
            out.append(",")
        out.append(token)

        if token_kind != "other":
            last_token_kind = token_kind
        idx = next_idx

    return "".join(out)


def _remove_trailing_commas(text: str) -> str:
    out: list[str] = []
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch != ",":
            out.append(ch)
            idx += 1
            continue

        lookahead = idx + 1
        while lookahead < len(text) and text[lookahead].isspace():
            lookahead += 1
        if lookahead < len(text) and text[lookahead] in "}]":
            idx += 1
            continue
        out.append(ch)
        idx += 1
    return "".join(out)


def repair_json_text(text: str) -> str:
    candidate = extract_balanced_json_candidate(strip_code_fences(text))
    candidate = _insert_missing_commas(candidate)
    candidate = _remove_trailing_commas(candidate)
    return candidate


def parse_json_value(text: str) -> Any:
    raw = strip_code_fences(text)
    candidate = extract_balanced_json_candidate(raw)
    last_err: Exception | None = None
    for attempt in (candidate, repair_json_text(raw)):
        try:
            return json.loads(attempt)
        except JSONDecodeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    raise ValueError("No JSON object found in model output")


def parse_json_object(text: str) -> dict[str, Any]:
    obj = parse_json_value(text)
    if not isinstance(obj, dict):
        raise ValueError("JSON output must be an object")
    return obj
