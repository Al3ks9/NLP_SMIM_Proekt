"""
Shared LLM call wrapper: 'ollama' (local, no limits) and 'openrouter' (remote,
quota-limited) backends behind one call(model, prompt, backend=...) interface.

Extracted from llm_probe.py so production code (llm_style_transfer.py) has an
honest home to depend on, instead of importing from a module whose own
docstring calls itself a throwaway diagnostic. llm_probe.py imports call()
from here and keeps everything probe-specific (script/infill checks, the
probe battery) to itself.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
OLLAMA_URL = os.environ.get('OLLAMA_HOST', 'http://localhost:11434') + '/api/chat'


def _post(url: str, body: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode('utf-8'),
        headers={'Content-Type': 'application/json', **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode("utf-8", "replace")[:400]}')
    except urllib.error.URLError as e:
        raise RuntimeError(f'Cannot reach {url}: {e.reason}')


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


def _api_key() -> str:
    """OPENROUTER_API_KEY from the environment, falling back to a .env file at ROOT."""
    key = os.environ.get('OPENROUTER_API_KEY')
    if key:
        return key

    env_file = ROOT / '.env'
    if env_file.exists():
        for line in env_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line.startswith('OPENROUTER_API_KEY'):
                return line.split('=', 1)[1].strip().strip('"\'')

    sys.exit(
        'OPENROUTER_API_KEY not found.\n'
        f'Either export it, or create {env_file} containing:\n'
        '    OPENROUTER_API_KEY=sk-or-...'
    )


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


BACKENDS = {'ollama': call_ollama, 'openrouter': call_openrouter}


def call(model: str, prompt: str, backend: str = 'ollama', **kw) -> str:
    return BACKENDS[backend](model, prompt, **kw)
