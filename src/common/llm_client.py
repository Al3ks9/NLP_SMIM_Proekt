"""
Shared LLM call wrapper: 'ollama' (local, no limits), 'openrouter' (remote,
quota-limited) and 'google' (Gemini API, direct) backends behind one
call(model, prompt, backend=...) interface.

Extracted from llm_probe.py so production code (llm_style_transfer.py) has an
honest home to depend on, instead of importing from a module whose own
docstring calls itself a throwaway diagnostic. llm_probe.py imports call()
from here and keeps everything probe-specific (script/infill checks, the
probe battery) to itself.
"""

import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger('llm_client')

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
OLLAMA_URL = os.environ.get('OLLAMA_HOST', 'http://localhost:11434') + '/api/chat'
GOOGLE_URL = 'https://generativelanguage.googleapis.com/v1beta/models'

# 408/429/5xx are transient (rate limit, upstream overload, timeout) and worth
# retrying; other 4xx (400 bad request, 403 bad key) mean every retry would
# fail identically, so those raise immediately instead of burning quota.
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds; doubles each attempt, so 1/2/4/8/16 before jitter


def _post(url: str, body: dict, headers: dict, timeout: int,
         max_retries: int = MAX_RETRIES) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode('utf-8'),
        headers={'Content-Type': 'application/json', **headers},
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = f'HTTP {e.code}: {e.read().decode("utf-8", "replace")[:400]}'
            if e.code not in RETRYABLE_STATUSES or attempt == max_retries:
                raise RuntimeError(detail)
        except urllib.error.URLError as e:
            detail = f'Cannot reach {url}: {e.reason}'
            if attempt == max_retries:
                raise RuntimeError(detail)

        # Exponential backoff with jitter so concurrent callers don't retry in lockstep.
        delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, BASE_DELAY)
        log.warning('%s -- retrying in %.1fs (attempt %d/%d)',
                   detail, delay, attempt + 1, max_retries)
        time.sleep(delay)


def _strip_think(text: str) -> str:
    """Qwen3 is a hybrid reasoning model; drop any thinking block that leaks through."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def call_ollama(model: str, prompt: str, temperature: float = 0.3,
                max_tokens: int = 700, num_ctx: int = 8192) -> str:
    """
    One chat completion from a local Ollama server.

    think=False matters: Qwen3 reasons by default, which would bury the output
    in a <think> block and make every call an order of magnitude slower.
    """
    payload = _post(OLLAMA_URL, {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'think': False,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
            'num_ctx': num_ctx,
        },
    }, headers={}, timeout=600)

    if 'message' not in payload:
        raise RuntimeError(f'Unexpected response: {json.dumps(payload)[:400]}')
    return _strip_think(payload['message'].get('content') or '')


def _key_from_env(names: tuple, example: str) -> str:
    """The first of `names` set in the environment, falling back to a .env file
    at ROOT. Exits with the variable name and an example rather than raising, so
    a missing key reads as setup instructions and not as a stack trace."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]

    env_file = ROOT / '.env'
    if env_file.exists():
        for line in env_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            for name in names:
                if line.startswith(name):
                    return line.split('=', 1)[1].strip().strip('"\'')

    sys.exit(
        f'{names[0]} not found.\n'
        f'Either export it, or create {env_file} containing:\n'
        f'    {names[0]}={example}'
    )


def _api_key() -> str:
    """OPENROUTER_API_KEY from the environment, falling back to a .env file at ROOT."""
    return _key_from_env(('OPENROUTER_API_KEY',), 'sk-or-...')


def _google_api_key() -> str:
    """GOOGLE_API_KEY (or GEMINI_API_KEY) from the environment or ROOT/.env."""
    return _key_from_env(('GOOGLE_API_KEY', 'GEMINI_API_KEY'), 'AIza...')


def call_openrouter(model: str, prompt: str, temperature: float = 0.3,
                    max_tokens: int = 700) -> str:
    """One chat completion from OpenRouter. Counts against the free daily quota."""
    key = _api_key()

    payload = _post(OPENROUTER_URL, {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': temperature,
        'max_tokens': max_tokens,
    }, headers={'Authorization': f'Bearer {key}'}, timeout=180)

    if 'choices' not in payload:
        raise RuntimeError(f'No choices in response: {json.dumps(payload)[:400]}')
    return _strip_think(payload['choices'][0]['message'].get('content') or '')


# Gemma 4 always reasons before answering, and the scratchpad comes back as a
# response part flagged thought=True. It cannot be turned off — a
# generationConfig.thinkingConfig.thinkingBudget is rejected with
# 400 "Thinking budget is not supported for this model" — and the thinking
# tokens are charged against maxOutputTokens. Hence the part filter below, a
# max_tokens default at the model's ceiling, and a retry.
#
# Thinking length is wildly variable for the same prompt, so no budget alone is
# safe: one style-transfer prompt spent 3231 thought tokens and 305 on the poem;
# a second spent 15997 and returned finishReason=MAX_TOKENS with no answer at
# all; the same prompt re-sent at 32768 then spent 2075. That is variance, not
# prompt difficulty, so a starved call is simply retried rather than budgeted
# around. Unused budget costs nothing — only generated tokens are billed.
def call_google(model: str, prompt: str, temperature: float = 0.3,
                max_tokens: int = 32768, attempts: int = 2) -> str:
    """
    One generateContent call against Google's Generative Language API, retried
    up to `attempts` times if thinking consumes the whole token budget.

    Used instead of the same model via OpenRouter because every :free Gemma
    route there answers 429 from a shared upstream pool; a direct Google key
    gets its own quota. Model ids are bare here ('gemma-4-31b-it'), not
    OpenRouter's vendor-prefixed form.
    """
    key = _google_api_key()
    name = model.split('models/', 1)[-1]  # accept bare or 'models/'-prefixed ids

    for attempt in range(1, attempts + 1):
        payload = _post(f'{GOOGLE_URL}/{name}:generateContent?key={key}', {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': temperature,
                'maxOutputTokens': max_tokens,
            },
        }, headers={}, timeout=600)

        candidates = payload.get('candidates') or []
        if not candidates:
            # Blocked or malformed rather than starved — retrying burns quota
            # for the same result.
            raise RuntimeError(f'No candidates in response: {json.dumps(payload)[:400]}')

        parts = candidates[0].get('content', {}).get('parts') or []
        text = ''.join(p.get('text', '') for p in parts if not p.get('thought')).strip()
        if text:
            return _strip_think(text)

        # Candidates but no answer part: the budget went entirely on thinking.
        # Returning '' here would look like a refusal instead of a starved call.
        finish = candidates[0].get('finishReason')
        thoughts = payload.get('usageMetadata', {}).get('thoughtsTokenCount')
        if attempt < attempts:
            log.info('call_google: no answer part (finishReason=%s, thoughts=%s) — '
                     'retrying %d/%d', finish, thoughts, attempt + 1, attempts)

    raise RuntimeError(
        f'No answer part after {attempts} attempt(s) (finishReason={finish!r}, '
        f'thoughtsTokenCount={thoughts}). Thinking consumed the whole '
        f'{max_tokens}-token budget.'
    )


BACKENDS = {'ollama': call_ollama, 'openrouter': call_openrouter, 'google': call_google}


def call(model: str, prompt: str, backend: str = 'ollama', **kw) -> str:
    return BACKENDS[backend](model, prompt, **kw)
