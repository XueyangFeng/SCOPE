import platform
import argparse
import time
import json
import re
import os
import shutil
import logging
import traceback
import threading
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service

from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY
from openai import OpenAI
from utils import get_web_element_rect, encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_proxy_env():
    """Remove system-level proxy env vars so API clients (httpx, openai) don't
    accidentally route through a SOCKS/HTTP proxy they can't handle."""
    for var in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
                'ALL_PROXY', 'all_proxy'):
        os.environ.pop(var, None)


def get_local_chrome_paths():
    """Return (chrome_binary, chromedriver_binary) from .local_browser/ if present."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.local_browser')
    chrome = os.path.join(base, 'chrome-linux64', 'chrome')
    driver = os.path.join(base, 'chromedriver-linux64', 'chromedriver')
    chrome = chrome if os.path.isfile(chrome) else None
    driver = driver if os.path.isfile(driver) else None
    return chrome, driver


def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Driver / Client creation
# ---------------------------------------------------------------------------

def driver_config(args, download_dir=None):
    options = webdriver.ChromeOptions()

    if args.save_accessibility_tree:
        args.force_device_scale = True

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if args.headless:
        options.add_argument("--headless=new")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-sync")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-crash-reporter")
    options.add_argument("--disable-breakpad")
    options.add_argument("--js-flags=--max-old-space-size=512")

    prefs = {
        "download.default_directory": download_dir or args.download_dir,
        "plugins.always_open_pdf_externally": True
    }
    if args.text_only:
        prefs["profile.managed_default_content_settings.images"] = 2  # disable images to save proxy traffic
    options.add_experimental_option("prefs", prefs)
    return options


def create_openai_client(api_key, base_url):
    import httpx
    # Bypass system proxy env vars (e.g. ALL_PROXY=socks5h://) which httpx doesn't support
    http_client = httpx.Client(trust_env=False)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text):
    if it == 1:
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        init_msg_format = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': init_msg},
            ]
        }
        init_msg_format['content'].append({"type": "image_url",
                                           "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        return curr_msg


def format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree):
    if it == 1:
        init_msg_format = {
            'role': 'user',
            'content': init_msg + '\n' + ac_tree
        }
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
            }
        return curr_msg


# ---------------------------------------------------------------------------
# GPT API call
# ---------------------------------------------------------------------------

def call_gpt4v_api(args, openai_client, messages):
    retry_times = 0
    kwargs = dict(model=args.api_model, messages=messages, max_tokens=1000)
    if args.seed is not None:
        kwargs['seed'] = args.seed
    if hasattr(args, 'temperature') and args.temperature is not None:
        kwargs['temperature'] = args.temperature

    while True:
        try:
            if not args.text_only:
                logging.info('Calling gpt4v API...')
                openai_response = openai_client.chat.completions.create(**kwargs)
            else:
                logging.info('Calling gpt4 API...')
                openai_response = openai_client.chat.completions.create(**kwargs, timeout=30)

            prompt_tokens = openai_response.usage.prompt_tokens
            completion_tokens = openai_response.usage.completion_tokens

            logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')

            gpt_call_error = False
            return prompt_tokens, completion_tokens, gpt_call_error, openai_response

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')

            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)

            elif type(e).__name__ == 'APIError':
                time.sleep(15)

            elif type(e).__name__ == 'InvalidRequestError':
                gpt_call_error = True
                return None, None, gpt_call_error, None

            else:
                gpt_call_error = True
                return None, None, gpt_call_error, None

        retry_times += 1
        if retry_times == 10:
            logging.info('Retrying too many times')
            return None, None, True, None


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    try:
        web_ele.clear()
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)

    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs


def exec_action_scroll(info, web_eles, driver_task, args, obs_info):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {args.window_height*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-args.window_height*2//3});")
    else:
        if not args.text_only:
            scroll_ele_number = int(scroll_ele_number)
            web_ele = web_eles[scroll_ele_number]
        else:
            element_box = obs_info[scroll_ele_number]['union_bound']
            element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
            web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(3)


# ---------------------------------------------------------------------------
# DCRL prefix replay, ID repair, validation, guided think
# ---------------------------------------------------------------------------

_PREFIX_REPAIR_CLIENT = None
_GUIDED_THINK_CLIENT = None


def _get_dcrl_client(api_base_env='EVAL_BASE_URL', api_key_env='EVAL_API_KEY',
                     fallback_base=None, fallback_key=None):
    """Create an OpenAI client for DCRL auxiliary calls (repair, guided think).
    Uses EVAL_BASE_URL by default (chatanywhere), NOT DCRL_BASE_URL (vLLM)."""
    import httpx
    api_key = (os.environ.get(api_key_env, '').strip()
               or os.environ.get('DCRL_API_KEY', '').strip()
               or fallback_key or 'EMPTY')
    api_base = (os.environ.get(api_base_env, '').strip()
                or fallback_base or 'https://api.openai.com/v1')
    return OpenAI(api_key=api_key, base_url=api_base,
                  http_client=httpx.Client(trust_env=False))


def load_prefix_data(prefix_root, task_id, iter_id, run_id):
    """Load prefix data from a previous DCRL run.
    Reads selected_anchor.jsonl to find which prefix was selected,
    then loads prefix_memory.json to get the actual actions.
    Returns dict with 'actions' and metadata, or None."""
    if not prefix_root or run_id is None or run_id <= 0:
        return None

    task_mem_dir = os.path.join(prefix_root, str(task_id))

    # Read selected anchor
    anchor_file = os.path.join(task_mem_dir, 'selected_anchor.jsonl')
    if not os.path.isfile(anchor_file):
        # Also try legacy path: selected_prefix/<task_id>.json
        legacy_file = os.path.join(prefix_root, 'selected_prefix', f'{task_id}.json')
        if os.path.isfile(legacy_file):
            try:
                with open(legacy_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                actions = data.get('actions') or data.get('prefix') or []
                if actions and 'ANSWER' in actions[-1].upper():
                    actions = actions[:-1]
                if not actions:
                    return None
                data['actions'] = actions
                return data
            except Exception as e:
                logging.warning(f'Failed to load legacy prefix: {e}')
        return None

    try:
        # Read last line of selected_anchor.jsonl
        with open(anchor_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return None
        anchor = json.loads(lines[-1])
        selected_prefix_id = anchor.get('selected_prefix_id', '')
        if not selected_prefix_id or selected_prefix_id == '-1':
            return None  # No prefix selected (restart from scratch)

        # Load prefix_memory.json to get actual actions
        prefix_mem_file = os.path.join(task_mem_dir, 'prefix_memory.json')
        if not os.path.isfile(prefix_mem_file):
            return None
        with open(prefix_mem_file, 'r', encoding='utf-8') as f:
            prefix_mem = json.load(f)

        # Find the selected prefix
        selected = None
        for p in prefix_mem.get('prefixes', []):
            if str(p.get('prefix_id')) == str(selected_prefix_id):
                selected = p
                break
        if not selected:
            return None

        # Collect all actions: prefix_actions (from parents) + own actions
        actions = []
        for item in selected.get('prefix_actions', []):
            act = item.get('action', '') if isinstance(item, dict) else str(item)
            act = act.strip()
            if act:
                actions.append(act)
        for item in selected.get('actions', []):
            act = item.get('action', '') if isinstance(item, dict) else str(item)
            act = act.strip()
            if act:
                actions.append(act)

        # Strip trailing ANSWER
        if actions and 'ANSWER' in actions[-1].upper():
            actions = actions[:-1]
        if not actions:
            return None

        # Also collect source observations for ID repair
        source_observations = {}
        for item in selected.get('prefix_actions', []) + selected.get('actions', []):
            if isinstance(item, dict) and 'observation_text' in item:
                idx = len(source_observations)
                source_observations[str(idx)] = item['observation_text']

        return {
            'actions': actions,
            'source_observations': source_observations,
            'selected_prefix_id': selected_prefix_id,
            'reference_url': selected.get('end_url', ''),
            'anchor': anchor,
        }
    except Exception as e:
        logging.warning(f'Failed to load prefix for task {task_id}: {e}')
        return None


def load_prefix_runtime_context(prefix_root, task_id):
    """Load failure context for guided think.
    Reads failure_memory.json from memory_store/<task_id>/."""
    task_mem_dir = os.path.join(prefix_root, str(task_id))

    # Try failure_memory.json (standard DCRL format)
    fm_file = os.path.join(task_mem_dir, 'failure_memory.json')
    if os.path.isfile(fm_file):
        try:
            with open(fm_file, 'r', encoding='utf-8') as f:
                fm = json.load(f)
            failures = fm.get('failures', [])
            if not failures:
                return {}
            # Build failure_memories text
            chunks = []
            for idx, item in enumerate(failures, start=1):
                summary = str(item.get('failure_summary', '')).strip()
                if summary:
                    chunks.append(f'[Failure {idx}]\nFailure summary: {summary}')
            failure_memories = '\n\n'.join(chunks) if chunks else ''
            # Get original failed behavior (first action of first failure CMU)
            orig_failed = ''
            for item in failures:
                for cmu in item.get('failure_cmus', []):
                    actions = cmu.get('actions', [])
                    if actions:
                        act = actions[0].get('action', '') if isinstance(actions[0], dict) else str(actions[0])
                        if act.strip():
                            orig_failed = act.strip()
                            break
                if orig_failed:
                    break
            return {
                'failure_memories': failure_memories,
                'original_failed_behavior': orig_failed,
            }
        except Exception:
            pass

    # Fallback: try legacy prefix_runtime/<task_id>.json
    ctx_file = os.path.join(prefix_root, 'prefix_runtime', f'{task_id}.json')
    if os.path.isfile(ctx_file):
        try:
            with open(ctx_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# --- Element ID helpers (WebVoyager action format: Click [15], Type [3]; text) ---

def _wv_extract_element_id(action_str):
    """'Click [15]' -> '15', 'Scroll [WINDOW]; down' -> 'WINDOW'"""
    m = re.search(r'\[(\d+|WINDOW)\]', action_str)
    return m.group(1) if m else ''


def _wv_replace_element_id(action_str, new_id):
    """('Click [15]', '20') -> 'Click [20]'"""
    return re.sub(r'\[(\d+)\]', f'[{new_id}]', action_str, count=1)


def _wv_extract_element_semantic(ac_tree_text, element_id):
    """From '[15] searchbox "Search Amazon"' extract 'searchbox Search Amazon'"""
    if not element_id or element_id == 'WINDOW' or not ac_tree_text:
        return ''
    pattern = rf'\[{re.escape(element_id)}\]\s*(.*)'
    m = re.search(pattern, ac_tree_text)
    if not m:
        return ''
    line = m.group(1).strip()
    line = re.sub(r"'([^']*)'", r'\1', line)
    line = re.sub(r'"([^"]*)"', r'\1', line)
    return re.sub(r'\s+', ' ', line).strip()[:200]


def _normalize_semantic(text):
    return re.sub(r'\s+', ' ', (text or '').strip().lower())


def _keyword_tokens(text):
    stop = {'the', 'a', 'an', 'on', 'in', 'of', 'to', 'for', 'with', 'and', 'or',
            'under', 'left', 'right', 'top', 'bottom', 'middle', 'center',
            'field', 'button', 'option', 'section', 'input', 'dropdown'}
    return {t for t in re.findall(r'[a-zA-Z0-9_]+', text.lower()) if len(t) >= 2 and t not in stop}


def _wv_action_semantic_ok(action_str, ac_tree_text, expected_hint=''):
    eid = _wv_extract_element_id(action_str)
    if not eid or eid == 'WINDOW':
        return True
    current_sem = _normalize_semantic(_wv_extract_element_semantic(ac_tree_text, eid))
    if not current_sem:
        return False
    expected = _normalize_semantic(expected_hint)
    if not expected:
        return True
    if expected in current_sem or current_sem in expected:
        return True
    exp_tokens = _keyword_tokens(expected)
    cur_tokens = _keyword_tokens(current_sem)
    if not exp_tokens:
        return True
    return len(exp_tokens & cur_tokens) > 0


def _wv_collect_candidates(ac_tree_text, max_candidates=300):
    candidates = []
    seen = set()
    for m in re.finditer(r'\[(\d+)\]\s*(.*)', ac_tree_text):
        eid = m.group(1)
        if eid in seen:
            continue
        seen.add(eid)
        sem = m.group(2).strip()
        sem = re.sub(r"'([^']*)'", r'\1', sem)
        sem = re.sub(r'"([^"]*)"', r'\1', sem)
        sem = re.sub(r'\s+', ' ', sem).strip()
        if not sem:
            continue
        candidates.append({'id': eid, 'semantic': sem[:180]})
        if len(candidates) >= max_candidates:
            break
    return candidates


def _extract_json_object(text):
    text = (text or '').strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def repair_prefix_element(ac_tree_text, action_str, model, expected_hint='',
                          api_base=None, api_key=None):
    """Use LLM to repair element ID when semantic mismatch detected."""
    global _PREFIX_REPAIR_CLIENT
    trace = {'repair_model': model, 'status': '', 'old_id': '', 'new_id': '',
             'candidate_count': 0, 'expected_hint': expected_hint,
             'raw_response': '', 'latency_ms': 0}
    old_id = _wv_extract_element_id(action_str)
    trace['old_id'] = old_id
    if not old_id or old_id == 'WINDOW':
        trace['status'] = 'no_element_id'
        return action_str, trace
    old_sem = _wv_extract_element_semantic(ac_tree_text, old_id)
    candidates = _wv_collect_candidates(ac_tree_text)
    trace['candidate_count'] = len(candidates)
    if not candidates:
        trace['status'] = 'no_candidates'
        return action_str, trace
    if _PREFIX_REPAIR_CLIENT is None:
        _PREFIX_REPAIR_CLIENT = _get_dcrl_client(fallback_base=api_base, fallback_key=api_key)
    prompt = json.dumps({
        'task': 'Choose the best element_id for this action from candidates. Change only element.',
        'raw_action': action_str,
        'old_element_id': old_id, 'old_element_semantic': old_sem,
        'expected_element_hint': expected_hint,
        'candidates': candidates,
        'output_schema': {'element_id': 'string from candidates.id'},
    }, ensure_ascii=False)
    t0 = time.time()
    try:
        response = _PREFIX_REPAIR_CLIENT.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are a web action repairer. Only output JSON with key element_id. Do not change action type or argument.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.0, max_tokens=4096, top_p=1.0,
        )
        trace['latency_ms'] = int((time.time() - t0) * 1000)
        content = response.choices[0].message.content or ''
        trace['raw_response'] = content
    except Exception as e:
        trace['status'] = f'api_error: {e}'
        trace['latency_ms'] = int((time.time() - t0) * 1000)
        return action_str, trace
    obj = _extract_json_object(content)
    if not obj:
        trace['status'] = 'json_not_found'
        return action_str, trace
    new_id = str(obj.get('element_id', '')).strip()
    trace['new_id'] = new_id
    if not new_id or new_id == old_id:
        trace['status'] = 'empty_or_same'
        return action_str, trace
    if not _wv_extract_element_semantic(ac_tree_text, new_id):
        trace['status'] = 'new_id_not_in_tree'
        return action_str, trace
    # Only allow changing element ID
    old_clean = _wv_replace_element_id(action_str, '__X__')
    new_action = _wv_replace_element_id(action_str, new_id)
    new_clean = _wv_replace_element_id(new_action, '__X__')
    if old_clean != new_clean:
        trace['status'] = 'rejected_non_element_change'
        return action_str, trace
    trace['status'] = 'repaired'
    return new_action, trace


def validate_prefix_replay(driver, reference_url):
    """Check if prefix replay achieved the same page state."""
    current_url = driver.current_url
    result = {'ok': False, 'decision': 'fail', 'reason': '',
              'current_url': current_url, 'reference_url': reference_url or ''}
    if not reference_url:
        result.update(ok=True, decision='pass', reason='no reference URL')
        return result
    from urllib.parse import urlparse
    cur = urlparse(current_url)
    ref = urlparse(reference_url)
    if cur.netloc == ref.netloc and cur.path.rstrip('/') == ref.path.rstrip('/'):
        result.update(ok=True, decision='pass', reason='URL host+path match')
        return result
    if cur.netloc == ref.netloc:
        result.update(decision='repairable',
                      reason=f'Same host, different path: {cur.path} vs {ref.path}')
        return result
    result['reason'] = f'URL mismatch: {current_url} vs {reference_url}'
    return result


def build_guided_think(task_ques, current_ac_tree, failure_memories='',
                       original_failed_behavior='', model='deepseek-v3.2',
                       api_base=None, api_key=None, timeout=20.0):
    """Call LLM to generate guided thought + action after prefix replay.
    Returns (thought, next_action, trace_dict).
    thought: natural reasoning (looks like agent's own Thought, no mention of failures/oracle).
    next_action: concrete WebVoyager action like 'Click [15]'."""
    global _GUIDED_THINK_CLIENT
    trace = {'guided_called': True, 'guided_model': model, 'guided_error': '',
             'raw_response': '', 'latency_ms': 0}
    if _GUIDED_THINK_CLIENT is None:
        _GUIDED_THINK_CLIENT = _get_dcrl_client(fallback_base=api_base, fallback_key=api_key)
    ac_tree_trimmed = (current_ac_tree or '')[:8000]
    prompt = f"""You are a web browsing agent reasoning about the next step.
A reusable action prefix has been replayed. You are now at the prefix frontier.
Based on the task, current page, failure reflections, and original failed behavior,
generate a better reasoning and next action.

Action format:
- Click [element_id]
- Type [element_id]; [content]
- Scroll [element_id or WINDOW]; [up or down]
- GoBack
- Google
- Wait
- ANSWER; [answer text]

Output JSON only:
{{
  "rationale": "internal analysis (NOT shown to the agent, can mention failures)",
  "thought": "natural reasoning grounded in the current page. Do NOT mention failures, history, oracle, or use words like 'avoid' or 'instead'. Write as if the agent is reasoning for the first time.",
  "next_action": "a concrete action like Click [15] or Type [3]; search term"
}}

Task: {task_ques}

Current page accessibility tree:
{ac_tree_trimmed}

Failure memories for this prefix:
{failure_memories if failure_memories else "(none)"}

Original failed behavior:
{original_failed_behavior if original_failed_behavior else "(none)"}
"""
    t0 = time.time()
    try:
        response = _GUIDED_THINK_CLIENT.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are a web action guide generator. Return JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.0, max_tokens=4096, top_p=1.0, timeout=timeout,
        )
        trace['latency_ms'] = int((time.time() - t0) * 1000)
        content = response.choices[0].message.content or ''
        trace['raw_response'] = content
    except Exception as e:
        trace['latency_ms'] = int((time.time() - t0) * 1000)
        trace['guided_error'] = repr(e)
        return '', '', trace
    obj = _extract_json_object(content)
    if not obj:
        trace['guided_error'] = 'json_not_found'
        return '', '', trace
    thought = str(obj.get('thought', '')).strip()
    next_action = str(obj.get('next_action', '')).strip()
    trace['thought'] = thought
    trace['next_action'] = next_action
    return thought, next_action, trace


def _build_prefilled_thought(thought, next_action=''):
    """Build prefilled assistant message for continue_final_message injection.
    E.g. 'Thought: I should search...\nAction: Type ['"""
    action_prefix = ''
    if next_action:
        m = re.match(r'((?:Click|Type|Scroll|GoBack|Google|Wait|ANSWER)\s*\[?)', next_action)
        if m:
            action_prefix = m.group(1)
    return f"Thought: {thought}\nAction: {action_prefix}"


def _format_guided_replay_output(thought, next_action):
    """Format as complete Thought+Action for direct replay / fallback."""
    return f"Thought: {thought}\nAction: {next_action}"


def _is_action_valid_for_tree(action_str, ac_tree_text):
    """Check if the element referenced by action exists in the current accessibility tree."""
    eid = _wv_extract_element_id(action_str)
    if not eid or eid == 'WINDOW':
        return True  # WINDOW scroll, GoBack, etc. always valid
    return bool(_wv_extract_element_semantic(ac_tree_text, eid))


def _execute_replay_action(driver, action_str, args, ac_tree, obs_info, task, task_dir):
    """Execute a single WebVoyager action during prefix replay."""
    # Strip Thought:/Action: prefixes if present
    if 'Action:' in action_str:
        parts = re.split(r'Thought:|Action:|Observation:', action_str)
        action_str = parts[2].strip() if len(parts) > 2 else parts[-1].strip()

    action_key, info = extract_information(action_str)
    if action_key is None:
        logging.warning(f'Prefix replay: could not parse action: {action_str}')
        return

    if action_key == 'click':
        click_id = info[0]
        element_box = obs_info[click_id]['union_bound']
        center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
        web_ele = driver.execute_script(
            "return document.elementFromPoint(arguments[0], arguments[1]);", center[0], center[1])
        exec_action_click(info, web_ele, driver)
    elif action_key == 'type':
        type_id = info['number']
        element_box = obs_info[type_id]['union_bound']
        center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
        web_ele = driver.execute_script(
            "return document.elementFromPoint(arguments[0], arguments[1]);", center[0], center[1])
        exec_action_type(info, web_ele, driver)
    elif action_key == 'scroll':
        exec_action_scroll(info, None, driver, args, obs_info)
    elif action_key == 'goback':
        driver.back()
        time.sleep(2)
    elif action_key == 'google':
        driver.get('https://www.google.com/')
        time.sleep(2)
    elif action_key == 'wait':
        time.sleep(5)
    elif action_key == 'answer':
        pass  # Don't execute ANSWER during prefix replay


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(task, args_dict, result_dir, api_key, base_url):
    """Run a single WebVoyager task. Designed to be called from ProcessPoolExecutor."""
    args = argparse.Namespace(**args_dict)
    start_time = datetime.now()
    task_dir = os.path.join(result_dir, 'task{}'.format(task["id"]))

    # Skip if already completed
    if os.path.isdir(task_dir) and os.path.isfile(os.path.join(task_dir, 'interact_messages.json')):
        return {"task_id": task["id"], "task_dir": task_dir, "total_cost": 0.0, "status": "skipped", "error": None}

    os.makedirs(task_dir, exist_ok=True)
    setup_logger(task_dir)
    logging.info(f'########## TASK{task["id"]} ##########')
    logging.info(f'Task start time: {start_time.isoformat()}')

    client = None
    driver_task = None
    local_proxy_proc = None
    local_proxy_log_fp = None
    messages = []
    accumulate_prompt_token = 0
    accumulate_completion_token = 0
    status = 'failed'
    error_message = None
    total_cost = 0.0

    try:
        client = create_openai_client(api_key, base_url)
        task_download_dir = os.path.join(args.download_dir, task["id"])
        os.makedirs(task_download_dir, exist_ok=True)

        # Start local SOCKS5 forward proxy (Chrome doesn't support auth in --proxy-server)
        local_proxy_port = None
        proxy_host = os.environ.get("PROXY_HOST", "")
        if proxy_host:
            import socket, subprocess
            proxy_port = os.environ.get("PROXY_PORT", "6754")
            proxy_user = os.environ.get("PROXY_USER", "")
            proxy_pass = os.environ.get("PROXY_PASS", "")
            upstream_proxy_host = os.environ.get("UPSTREAM_PROXY_HOST", "")
            upstream_proxy_port = os.environ.get("UPSTREAM_PROXY_PORT", "7890")
            # Find a free local port
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0))
                local_proxy_port = s.getsockname()[1]
            local_proxy_script = os.path.join(os.path.dirname(__file__), 'local_proxy.py')
            local_proxy_log_path = os.path.join(task_dir, 'local_proxy.log')
            local_proxy_log_fp = open(local_proxy_log_path, 'a', encoding='utf-8')
            logging.info(
                f'Local proxy chain: chrome -> socks5://127.0.0.1:{local_proxy_port} '
                f'-> {proxy_host}:{proxy_port}'
                + (f' via {upstream_proxy_host}:{upstream_proxy_port}' if upstream_proxy_host else '')
            )
            logging.info(f'Local proxy log: {local_proxy_log_path}')
            local_proxy_proc = subprocess.Popen(
                ['python', '-u', local_proxy_script, str(local_proxy_port),
                 proxy_host, proxy_port, proxy_user, proxy_pass],
                stdout=local_proxy_log_fp, stderr=subprocess.STDOUT
            )
            # Wait until local proxy is actually listening (up to 10s)
            for _ in range(20):
                try:
                    with socket.socket() as _s:
                        _s.settimeout(0.5)
                        _s.connect(('127.0.0.1', local_proxy_port))
                    break
                except (ConnectionRefusedError, OSError):
                    time.sleep(0.5)
            else:
                logging.warning('local_proxy did not start in time, proceeding anyway')

        options = driver_config(args, download_dir=task_download_dir)
        if local_proxy_port:
            options.add_argument(f"--proxy-server=socks5://127.0.0.1:{local_proxy_port}")

        chrome_path, driver_path = get_local_chrome_paths()
        if chrome_path:
            options.binary_location = chrome_path
        if driver_path:
            service = Service(executable_path=driver_path)
            driver_task = webdriver.Chrome(service=service, options=options)
        else:
            driver_task = webdriver.Chrome(options=options)

        driver_task.set_window_size(args.window_width, args.window_height)
        driver_task.get(task['web'])
        try:
            driver_task.find_element(By.TAG_NAME, 'body').click()
        except:
            pass
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        time.sleep(5)

        # Clean up download dir
        for filename in os.listdir(task_download_dir):
            file_path = os.path.join(task_download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        download_files = []

        fail_obs = ""
        pdf_obs = ""
        warn_obs = ""
        pattern = r'Thought:|Action:|Observation:'

        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
        if args.text_only:
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT_TEXT_ONLY}]
            obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."

        init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
        init_msg = init_msg.replace('https://www.example.com', task['web'])
        init_msg = init_msg + obs_prompt

        # DCRL prefix replay + guided think
        prefix_data = None
        prefix_replayed = False
        guided_thought = ''
        guided_action = ''
        guided_api_base = getattr(args, 'guided_api_base', None)
        guided_api_key = getattr(args, 'guided_api_key', None)

        if hasattr(args, 'prefix_root') and args.prefix_root:
            iter_id = getattr(args, 'iter', None)
            run_id = getattr(args, 'run', None)
            prefix_data = load_prefix_data(args.prefix_root, task['id'], iter_id, run_id)

        # --- Prefix replay (if non-empty prefix exists) ---
        if prefix_data and prefix_data.get('actions'):
            prefix_actions = prefix_data.get('actions', [])
            source_obs = prefix_data.get('source_observations', {})
            logging.info(f'Replaying prefix with {len(prefix_actions)} actions')

            id_repair_enable = getattr(args, 'prefix_id_repair_enable', False)
            id_repair_model = getattr(args, 'prefix_id_repair_model', 'deepseek-v3.2')

            replay_result = {'success': True, 'steps_completed': 0, 'repair_traces': [],
                             'last_ac_tree': '', 'last_obs_info': {}}
            for pidx, p_action in enumerate(prefix_actions):
                try:
                    p_ac_path = os.path.join(task_dir, f'prefix_tree_{pidx}')
                    p_ac_tree, p_obs_info = get_webarena_accessibility_tree(driver_task, p_ac_path)
                except Exception as e:
                    logging.error(f'Prefix step {pidx}: failed to get tree: {e}')
                    replay_result['success'] = False
                    break

                replay_action = p_action
                p_eid = _wv_extract_element_id(p_action)
                if id_repair_enable and p_eid and p_eid != 'WINDOW':
                    expected_hint = ''
                    if source_obs and pidx > 0:
                        prev_obs = source_obs.get(str(pidx - 1), '')
                        if prev_obs:
                            expected_hint = _wv_extract_element_semantic(prev_obs, p_eid)
                    if not _wv_action_semantic_ok(p_action, p_ac_tree, expected_hint):
                        logging.info(f'Prefix step {pidx}: element [{p_eid}] mismatch, repairing')
                        try:
                            replay_action, repair_trace = repair_prefix_element(
                                p_ac_tree, p_action, id_repair_model,
                                expected_hint=expected_hint,
                                api_base=guided_api_base, api_key=guided_api_key)
                            if repair_trace.get('status') == 'repaired':
                                logging.info(f'Prefix repair {pidx}: {p_action} -> {replay_action}')
                            replay_result['repair_traces'].append({'step': pidx, **repair_trace})
                        except Exception as e:
                            logging.warning(f'Prefix repair {pidx} exception: {e}')

                try:
                    driver_task.save_screenshot(os.path.join(task_dir, f'prefix_screenshot_{pidx}.png'))
                except Exception:
                    pass

                try:
                    _execute_replay_action(driver_task, replay_action, args, p_ac_tree, p_obs_info, task, task_dir)
                except Exception as e:
                    logging.error(f'Prefix step {pidx} execution failed: {e}')
                    replay_result['success'] = False
                    break
                replay_result['steps_completed'] = pidx + 1

            # Prefix validation
            prefix_validate = getattr(args, 'prefix_validate_replay', False)
            on_fail = getattr(args, 'prefix_validate_on_fail', 'restart')
            restart_from_scratch = False
            if prefix_validate and replay_result['success']:
                ref_url = prefix_data.get('reference_url', '')
                val = validate_prefix_replay(driver_task, ref_url)
                logging.info(f'Prefix validation: {val}')
                if not val.get('ok'):
                    logging.warning(f'Prefix validation failed: {val.get("reason")}')
                    if on_fail == 'restart':
                        restart_from_scratch = True
                        logging.info('Restarting task from scratch (prefix validation failed)')
                    elif on_fail == 'skip':
                        status = 'prefix_validation_skip'
                        print_message(messages, task_dir)
                        raise Exception('prefix_validation_skip')

            if restart_from_scratch:
                driver_task.get(task['web'])
                try:
                    driver_task.find_element(By.TAG_NAME, 'body').click()
                except:
                    pass
                time.sleep(5)
                prefix_replayed = False
            elif replay_result['success'] and replay_result['steps_completed'] > 0:
                prefix_replayed = True

        # --- Guided think (runs if failure memory exists, regardless of prefix) ---
        guided_mode = getattr(args, 'guided_think_mode', 'off')
        if guided_mode == 'auto' and hasattr(args, 'prefix_root') and args.prefix_root and getattr(args, 'run', 0) and getattr(args, 'run', 0) > 0:
            runtime_ctx = load_prefix_runtime_context(args.prefix_root, task['id'])
            failure_memories = runtime_ctx.get('failure_memories', '')
            orig_failed = runtime_ctx.get('original_failed_behavior', '')
            # Only call guided think if there's failure context to work with
            if failure_memories or orig_failed:
                try:
                    gt_ac_path = os.path.join(task_dir, 'guided_think_tree')
                    gt_ac_tree, _ = get_webarena_accessibility_tree(driver_task, gt_ac_path)
                except Exception:
                    gt_ac_tree = ''
                guided_model = getattr(args, 'guided_think_model', 'deepseek-v3.2')
                guided_timeout = getattr(args, 'guided_think_timeout', 20.0)
                try:
                    guided_thought, guided_action, guided_trace = build_guided_think(
                        task['ques'], gt_ac_tree, failure_memories, orig_failed,
                        model=guided_model,
                        api_base=guided_api_base, api_key=guided_api_key,
                        timeout=guided_timeout)
                    logging.info(f'Guided think: thought={guided_thought[:100]}, action={guided_action}')
                except Exception as e:
                    logging.warning(f'Guided think failed: {e}')

        it = 0
        while it < args.max_iter:
            logging.info(f'Iter: {it}')
            it += 1
            if not fail_obs:
                try:
                    if not args.text_only:
                        rects, web_eles, web_eles_text = get_web_element_rect(driver_task, fix_color=args.fix_box_color)
                    else:
                        accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                        ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                except Exception as e:
                    if not args.text_only:
                        logging.error('Driver error when adding set-of-mark.')
                    else:
                        logging.error('Driver error when obtaining accessibility tree.')
                    logging.error(e)
                    break

                try:
                    img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
                    driver_task.save_screenshot(img_path)
                except Exception as e:
                    logging.error(f'Screenshot failed: {e}')
                    break

                if (not args.text_only) and args.save_accessibility_tree:
                    try:
                        accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                        get_webarena_accessibility_tree(driver_task, accessibility_tree_path)
                    except Exception:
                        pass

                b64_img = encode_image(img_path)

                if not args.text_only:
                    curr_msg = format_msg(it, init_msg, pdf_obs, warn_obs, b64_img, web_eles_text)
                else:
                    curr_msg = format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree)
                messages.append(curr_msg)
            else:
                curr_msg = {
                    'role': 'user',
                    'content': fail_obs
                }
                messages.append(curr_msg)

            if not args.text_only:
                messages = clip_message_and_obs(messages, args.max_attached_imgs)
            else:
                messages = clip_message_and_obs_text_only(messages, args.max_attached_imgs)

            # Guided think injection on first post-prefix iteration
            guided_used_this_step = False
            if it == 1 and prefix_replayed and guided_thought and guided_action:
                apply_mode = getattr(args, 'guided_think_apply_mode', 'guideness')
                if apply_mode == 'replay':
                    # Replay mode: use guided output directly, skip model call
                    gpt_4v_res = _format_guided_replay_output(guided_thought, guided_action)
                    logging.info(f'Guided replay: {gpt_4v_res}')
                    messages.append({'role': 'assistant', 'content': gpt_4v_res})
                    guided_used_this_step = True
                    # Fake token counts
                    accumulate_prompt_token += 0
                    accumulate_completion_token += 0
                elif apply_mode == 'guideness':
                    # Guideness mode: prefill thought, let model complete via continue_final_message
                    prefill = _build_prefilled_thought(guided_thought, guided_action)
                    prefill_messages = messages + [{'role': 'assistant', 'content': prefill}]
                    try:
                        logging.info(f'Guided prefill: {prefill[:120]}')
                        kwargs_prefill = dict(
                            model=args.api_model, messages=prefill_messages,
                            max_tokens=1000, timeout=30,
                            extra_body={'continue_final_message': True, 'add_generation_prompt': False},
                        )
                        if args.seed is not None:
                            kwargs_prefill['seed'] = args.seed
                        if hasattr(args, 'temperature') and args.temperature is not None:
                            kwargs_prefill['temperature'] = args.temperature
                        prefill_response = client.chat.completions.create(**kwargs_prefill)
                        completion = prefill_response.choices[0].message.content or ''
                        gpt_4v_res = prefill + completion
                        accumulate_prompt_token += (prefill_response.usage.prompt_tokens or 0)
                        accumulate_completion_token += (prefill_response.usage.completion_tokens or 0)
                        logging.info(f'Guided prefill completed: {gpt_4v_res[:200]}')

                        # Fallback 1: check if completed action references a valid element
                        if args.text_only and not _is_action_valid_for_tree(gpt_4v_res, ac_tree):
                            logging.warning('Guided prefill action not valid for current tree, falling back to replay')
                            gpt_4v_res = _format_guided_replay_output(guided_thought, guided_action)

                        messages.append({'role': 'assistant', 'content': gpt_4v_res})
                        guided_used_this_step = True
                    except Exception as e:
                        logging.warning(f'Guided prefill failed ({e}), falling back to normal call')
                        # Fall through to normal API call below

            if not guided_used_this_step:
                prompt_tokens, completion_tokens, gpt_call_error, openai_response = call_gpt4v_api(args, client, messages)

                if gpt_call_error:
                    break
                else:
                    accumulate_prompt_token += prompt_tokens
                    accumulate_completion_token += completion_tokens
                    logging.info(f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
                    logging.info('API call complete...')
                gpt_4v_res = openai_response.choices[0].message.content
                messages.append({'role': 'assistant', 'content': gpt_4v_res})

            if (not args.text_only) and rects:
                logging.info(f"Num of interactive elements: {len(rects)}")
                for rect_ele in rects:
                    driver_task.execute_script("arguments[0].remove()", rect_ele)
                rects = []

            try:
                assert 'Thought:' in gpt_4v_res and 'Action:' in gpt_4v_res
            except AssertionError as e:
                logging.error(e)
                fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                continue

            chosen_action = re.split(pattern, gpt_4v_res)[2].strip()
            action_key, info = extract_information(chosen_action)

            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""
            try:
                window_handle_task = driver_task.current_window_handle
                driver_task.switch_to.window(window_handle_task)

                if action_key == 'click':
                    if not args.text_only:
                        click_ele_number = int(info[0])
                        web_ele = web_eles[click_ele_number]
                    else:
                        click_ele_number = info[0]
                        element_box = obs_info[click_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                              element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    ele_tag_name = web_ele.tag_name.lower()
                    ele_type = web_ele.get_attribute("type")

                    exec_action_click(info, web_ele, driver_task)

                    current_files = sorted(os.listdir(task_download_dir))
                    if current_files != download_files:
                        time.sleep(10)
                        current_files = sorted(os.listdir(task_download_dir))

                        current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in download_files and pdf_file.endswith('.pdf')]
                        if current_download_file:
                            pdf_file = current_download_file[0]
                            pdf_obs = get_pdf_retrieval_ans_from_assistant(client, os.path.join(task_download_dir, pdf_file), task['ques'])
                            shutil.copy(os.path.join(task_download_dir, pdf_file), task_dir)
                            pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                        download_files = current_files

                    if ele_tag_name == 'button' and ele_type == 'submit':
                        time.sleep(10)

                elif action_key == 'wait':
                    time.sleep(5)

                elif action_key == 'type':
                    if not args.text_only:
                        type_ele_number = int(info['number'])
                        web_ele = web_eles[type_ele_number]
                    else:
                        type_ele_number = info['number']
                        element_box = obs_info[type_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                              element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    warn_obs = exec_action_type(info, web_ele, driver_task)
                    if 'wolfram' in task['web']:
                        time.sleep(5)

                elif action_key == 'scroll':
                    if not args.text_only:
                        exec_action_scroll(info, web_eles, driver_task, args, None)
                    else:
                        exec_action_scroll(info, None, driver_task, args, obs_info)

                elif action_key == 'goback':
                    driver_task.back()
                    time.sleep(2)

                elif action_key == 'google':
                    driver_task.get('https://www.google.com/')
                    time.sleep(2)

                elif action_key == 'answer':
                    logging.info(info['content'])
                    logging.info('finish!!')
                    status = 'finished'
                    break

                else:
                    raise NotImplementedError
                fail_obs = ""
            except Exception as e:
                logging.error('driver error info:')
                logging.error(e)
                # Fallback 2: if guided step failed execution, try guided replay action
                if guided_used_this_step and guided_action:
                    logging.info('Guided step failed, falling back to guided replay action')
                    try:
                        fb_ac_tree, fb_obs_info = get_webarena_accessibility_tree(driver_task, None)
                        _execute_replay_action(driver_task, guided_action, args,
                                               fb_ac_tree, fb_obs_info, task, task_dir)
                        fail_obs = ""
                        logging.info(f'Guided fallback executed: {guided_action}')
                    except Exception as fb_e:
                        logging.warning(f'Guided fallback also failed: {fb_e}')
                        fail_obs = "The action you have chosen cannot be exected. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
                elif 'element click intercepted' not in str(e):
                    fail_obs = "The action you have chosen cannot be exected. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
                else:
                    fail_obs = ""
                time.sleep(2)

        print_message(messages, task_dir)

    except Exception as e:
        logging.error(f'Task {task["id"]} failed with exception: {e}')
        logging.error(traceback.format_exc())
        error_message = str(e)
    finally:
        if driver_task is not None:
            try:
                driver_task.quit()
            except Exception:
                logging.error('Failed to quit driver cleanly')
                logging.error(traceback.format_exc())
        if local_proxy_proc is not None:
            try:
                local_proxy_proc.terminate()
            except Exception:
                pass
        if local_proxy_log_fp is not None:
            try:
                local_proxy_log_fp.close()
            except Exception:
                pass
        total_cost = accumulate_prompt_token / 1000 * 0.01 + accumulate_completion_token / 1000 * 0.03
        end_time = datetime.now()
        logging.info(f'Task end time: {end_time.isoformat()}')
        logging.info(f'Task duration seconds: {(end_time - start_time).total_seconds():.2f}')
        logging.info(f'Task status: {status}')
        logging.info(f'Total cost: {total_cost}')

    return {"task_id": task["id"], "task_dir": task_dir, "total_cost": total_cost, "status": status, "error": error_message}


# ---------------------------------------------------------------------------
# Site-group runner (with inline eval)
# ---------------------------------------------------------------------------

def run_site_group(site_name, site_tasks, args_dict, result_dir, api_key, base_url, progress_queue=None):
    import concurrent.futures
    task_timeout = args_dict.get('task_timeout', 900)
    eval_model = args_dict.get('eval_model')
    site_results = []
    for task in site_tasks:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_task, task, args_dict, result_dir, api_key, base_url)
            try:
                result = future.result(timeout=task_timeout)
            except concurrent.futures.TimeoutError:
                logging.warning(f'Task {task["id"]} timed out after {task_timeout}s, skipping.')
                result = {"task_id": task["id"], "task_dir": None, "total_cost": 0.0, "status": "timeout", "error": "task_timeout"}
        # Inline eval after task completes
        if eval_model and result.get('task_dir'):
            try:
                import sys as _sys
                import os as _os
                _eval_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'evaluation')
                if _eval_dir not in _sys.path:
                    _sys.path.insert(0, _eval_dir)
                from auto_eval import auto_eval_by_gpt4v
                from openai import OpenAI as _OpenAI
                _eval_api_key = (_os.environ.get('EVAL_API_KEY')
                                 or _os.environ.get('OPENAI_API_KEY')
                                 or api_key)
                _eval_base_url = (_os.environ.get('EVAL_BASE_URL') or 'https://api.openai.com/v1')
                import httpx as _httpx
                _eval_client = _OpenAI(api_key=_eval_api_key, base_url=_eval_base_url,
                                       http_client=_httpx.Client(trust_env=False))
                auto_eval_by_gpt4v(result['task_dir'], _eval_client, eval_model,
                                   img_num=args_dict.get('max_attached_imgs', 1))
            except Exception as _e:
                logging.warning(f'Inline eval failed for task {task["id"]}: {_e}')
        site_results.append(result)
        if progress_queue is not None:
            progress_queue.put(result['task_id'])
    return site_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test.json')
    parser.add_argument('--task_id', type=int, default=None, help='Only run the specified 0-based task index from --test_file.')
    parser.add_argument('--task_ids', type=str, default=None, help='Comma-separated 0-based task indexes from --test_file.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes. Each worker uses its own browser instance.')
    parser.add_argument('--group_by_site', action='store_true', help='Parallelize by website. Each worker processes one site serially.')
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument("--api_key", default=None, type=str, help="OpenAI API key. Defaults to env OPENAI_API_KEY.")
    parser.add_argument("--api_model", default="gpt-4-vision-preview", type=str, help="api model name")
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--text_only", action='store_true')
    parser.add_argument("--headless", action='store_true', help='The window of selenium')
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)
    parser.add_argument("--fix_box_color", action='store_true')
    parser.add_argument("--save_chromedriver_log", action='store_true')
    parser.add_argument("--task_timeout", type=int, default=900,
                        help='Per-task timeout in seconds (default: 900 = 15min). Only applies with --group_by_site.')
    parser.add_argument("--eval_model", type=str, default=None,
                        help='If set, run auto-eval after each task using this model (e.g. gpt-4o). '
                             'Uses EVAL_API_KEY/EVAL_BASE_URL env vars, falling back to OPENAI_API_KEY.')
    parser.add_argument("--prefix_root", type=str, default=None,
                        help='Path to DCRL memory_store root directory. Enables prefix replay if set.')
    parser.add_argument("--prefix_validate_replay", action='store_true',
                        help='Validate prefix replay state matches reference.')
    parser.add_argument("--prefix_validate_on_fail", type=str, default='restart',
                        choices=['skip', 'restart', 'raise'],
                        help='Action on prefix validation failure (default: restart).')
    parser.add_argument("--prefix_id_repair_enable", action='store_true',
                        help='Enable LLM-based element ID repair during prefix replay.')
    parser.add_argument("--prefix_id_repair_model", type=str, default='deepseek-v3.2',
                        help='Model for prefix element ID repair.')
    parser.add_argument("--guided_think_mode", type=str, default='off',
                        choices=['off', 'auto'],
                        help='Guided think mode after prefix replay (default: off).')
    parser.add_argument("--guided_think_apply_mode", type=str, default='guideness',
                        choices=['guideness', 'replay'],
                        help='How to apply guided think: inject hint or replay action.')
    parser.add_argument("--guided_think_model", type=str, default='deepseek-v3.2',
                        help='Model for guided think.')
    parser.add_argument("--guided_api_base", type=str, default=None,
                        help='API base URL for guided think / ID repair.')
    parser.add_argument("--guided_api_key", type=str, default=None,
                        help='API key for guided think / ID repair.')
    parser.add_argument("--guided_think_timeout", type=float, default=20.0,
                        help='Timeout for guided think API call.')
    parser.add_argument("--iter", type=int, default=None,
                        help='DCRL iteration index (used with --prefix_root).')
    parser.add_argument("--run", type=int, default=None,
                        help='DCRL run index within iteration (used with --prefix_root).')

    args = parser.parse_args()

    _clear_proxy_env()

    # API key / base_url
    api_key = args.api_key or os.environ.get('DCRL_API_KEY') or os.environ.get('OPENAI_API_KEY') or 'EMPTY'
    base_url = args.base_url or os.environ.get('DCRL_BASE_URL') or os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'

    # Output directory (flat — no timestamp subdirectory)
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)

    # Load tasks
    tasks = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))

    # Optionally filter tasks
    if args.task_id is not None:
        tasks = [tasks[args.task_id]]
    elif args.task_ids:
        indices = [int(x.strip()) for x in args.task_ids.split(',')]
        tasks = [tasks[i] for i in indices]

    # Skip already-completed tasks
    completed_task_ids = set()
    if os.path.isdir(result_dir):
        for d in os.listdir(result_dir):
            msg_file = os.path.join(result_dir, d, 'interact_messages.json')
            if os.path.isfile(msg_file):
                tid = d.replace('task', '', 1)
                completed_task_ids.add(tid)
    if completed_task_ids:
        before = len(tasks)
        tasks = [t for t in tasks if str(t['id']) not in completed_task_ids]
        logging.info(f'Skipping {before - len(tasks)} already-completed tasks, {len(tasks)} remaining')
        print(f'Skipping {before - len(tasks)} already-completed tasks, {len(tasks)} remaining', flush=True)

    if not tasks:
        print('All tasks already completed.', flush=True)
        return

    args_dict = vars(args).copy()
    max_workers = min(args.num_workers, len(tasks))
    total_tasks = len(tasks)
    completed_tasks = 0

    if max_workers <= 1:
        for task in tasks:
            result = run_task(task, args_dict, result_dir, api_key, base_url)
            completed_tasks += 1
            print(f'[{completed_tasks}/{total_tasks}] finished {result["task_id"]}', flush=True)
    else:
        if args.group_by_site:
            grouped_tasks = defaultdict(list)
            for task in tasks:
                grouped_tasks[task["web_name"]].append(task)
            with Manager() as manager:
                progress_queue = manager.Queue()

                def _print_progress():
                    done = 0
                    while True:
                        task_id = progress_queue.get()
                        if task_id is None:
                            break
                        done += 1
                        print(f'[{done}/{total_tasks}] finished {task_id}', flush=True)

                printer = threading.Thread(target=_print_progress, daemon=True)
                printer.start()

                with ProcessPoolExecutor(max_workers=min(max_workers, len(grouped_tasks))) as executor:
                    futures = [
                        executor.submit(run_site_group, site_name, site_tasks, args_dict, result_dir, api_key, base_url, progress_queue)
                        for site_name, site_tasks in grouped_tasks.items()
                    ]
                    for future in as_completed(futures):
                        future.result()  # surface exceptions

                progress_queue.put(None)  # stop printer thread
                printer.join()
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(run_task, task, args_dict, result_dir, api_key, base_url)
                    for task in tasks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    completed_tasks += 1
                    print(f'[{completed_tasks}/{total_tasks}] finished {result["task_id"]}', flush=True)


if __name__ == '__main__':
    main()
    print('End of process')
