from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests


def _log(log_file: str, message: str) -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f'[{ts}] {message}\n')


def reset_web_backend(reset_url: str, status_url: str, ready_text: str, timeout_s: int, log_file: str) -> int:
    _log(log_file, f'reset: {reset_url}')
    try:
        resp = requests.get(reset_url, timeout=30)
        resp.raise_for_status()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(resp.text)
            if not resp.text.endswith('\n'):
                f.write('\n')
    except Exception:
        _log(log_file, 'reset request failed')
    _log(log_file, 'post-reset sleep 30s')
    time.sleep(30)

    deadline = time.time() + timeout_s
    next_log = time.time() + 20
    while time.time() < deadline:
        try:
            status_resp = requests.get(status_url, timeout=10).text
            if ready_text in status_resp:
                _log(log_file, f'ready: {status_url}')
                return 0
        except Exception:
            pass
        now = time.time()
        if now >= next_log:
            _log(log_file, f'waiting for ready: {status_url}')
            next_log = now + 20
        time.sleep(1)
    _log(log_file, f'backend not ready after {timeout_s}s: {status_url}')
    return 1


def restart_vllm_and_wait(log_file: str) -> int:
    timeout_s = int(os.environ['VLLM_RESTART_TIMEOUT'])
    base_url = os.environ['VLLM_RESTART_BASE_URL'].rstrip('/')
    status_url = f'{base_url}/status'
    health_url = f'{base_url}/health'
    ready_url = f"{os.environ['SERVER'].rstrip('/')}/models"
    restart_key = os.environ.get('VLLM_RESTART_KEY', '')
    restart_model = os.environ['VLLM_RESTART_MODEL']
    openai_api_key = os.environ.get('OPENAI_API_KEY', '')

    if not restart_key:
        _log(log_file, '[vllm] VLLM_RESTART_KEY is empty')
        return 1

    restart_url = f"{base_url}/restart?key={quote(restart_key, safe='')}&model={quote(restart_model, safe='')}"
    status_auth_url = f"{status_url}?key={quote(restart_key, safe='')}"

    _log(log_file, f'[vllm] restart: {restart_url}')
    try:
        requests.get(health_url, timeout=10)
    except Exception:
        pass

    try:
        resp = requests.post(restart_url, timeout=30)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(resp.text)
            if not resp.text.endswith('\n'):
                f.write('\n')
        if 200 <= resp.status_code < 300:
            pass
        elif resp.status_code == 409:
            _log(log_file, '[vllm] restart already in progress (409), continue waiting')
        else:
            _log(log_file, f'[vllm] restart failed, HTTP {resp.status_code}')
            return 1
    except Exception as exc:
        _log(log_file, f'[vllm] restart failed: {exc}')
        return 1

    deadline = time.time() + timeout_s
    next_log = time.time() + 15
    last_status = ''
    while time.time() < deadline:
        try:
            status_resp = requests.get(status_auth_url, timeout=10).text
            last_status = status_resp
            status_lower = status_resp.lower()
            if any(x in status_lower for x in ('ready', 'running', 'success', 'succeeded', 'done', 'idle', 'jiuxu')):
                _log(log_file, f'[vllm] status ready: {status_resp}')
                return 0
            if any(x in status_lower for x in ('fail', 'failed', 'error', 'stopped', 'timeout', 'crash', 'aborted', 'shibai')):
                _log(log_file, f'[vllm] status failed: {status_resp}')
                return 1
        except Exception:
            try:
                resp = requests.get(ready_url, headers={'Authorization': f'Bearer {openai_api_key}'}, timeout=10)
                resp.raise_for_status()
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(resp.text)
                    if not resp.text.endswith('\n'):
                        f.write('\n')
                _log(log_file, f'[vllm] ready (fallback): {ready_url}')
                return 0
            except Exception:
                pass
        now = time.time()
        if now >= next_log:
            if last_status:
                _log(log_file, f'[vllm] status: {last_status}')
            else:
                _log(log_file, f'[vllm] waiting for status: {status_auth_url}')
            next_log = now + 15
        time.sleep(1)

    _log(log_file, f'[vllm] not ready after {timeout_s}s: {status_auth_url}')
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_reset = sub.add_parser('reset-web')
    p_reset.add_argument('--reset-url', required=True)
    p_reset.add_argument('--status-url', required=True)
    p_reset.add_argument('--ready-text', required=True)
    p_reset.add_argument('--timeout', type=int, required=True)
    p_reset.add_argument('--log-file', required=True)

    p_vllm = sub.add_parser('restart-vllm')
    p_vllm.add_argument('--log-file', required=True)

    args = parser.parse_args()
    if args.cmd == 'reset-web':
        raise SystemExit(reset_web_backend(args.reset_url, args.status_url, args.ready_text, args.timeout, args.log_file))
    raise SystemExit(restart_vllm_and_wait(args.log_file))


if __name__ == '__main__':
    main()
