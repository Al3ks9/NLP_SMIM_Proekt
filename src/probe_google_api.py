"""
Connectivity probe for Gemma via Google's Generative Language API directly,
bypassing OpenRouter's shared free pool (which returns 429 "temporarily
rate-limited upstream" for every :free Gemma route).

Standalone and throwaway-ish: it answers "does this key reach a Gemma model
that can write Macedonian, and under what id?" before any of it is wired into
llm_client.py as a real backend.

Reads GOOGLE_API_KEY (or GEMINI_API_KEY) from the environment, falling back to
a .env file at the repo root.

Usage:
    uv run python src/probe_google_api.py                 # list gemma models, then probe the best one
    uv run python src/probe_google_api.py --list          # list only
    uv run python src/probe_google_api.py --model gemma-4-31b-it
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = 'https://generativelanguage.googleapis.com/v1beta'

# A real task, not "say hello": Macedonian generation is the whole reason for
# switching models, so the probe should fail loudly if the model can't do it.
PROBE_PROMPT = (
    'Write two lines of original Macedonian verse, in Cyrillic script, about '
    'a soldier buried under a willow. Reply with the two lines only.'
)


def api_key() -> str:
    for var in ('GOOGLE_API_KEY', 'GEMINI_API_KEY'):
        if os.environ.get(var):
            return os.environ[var]

    env_file = ROOT / '.env'
    if env_file.exists():
        for line in env_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            for var in ('GOOGLE_API_KEY', 'GEMINI_API_KEY'):
                if line.startswith(var):
                    return line.split('=', 1)[1].strip().strip('"\'')

    sys.exit(
        'GOOGLE_API_KEY not found.\n'
        f'Either export it, or add a line to {env_file}:\n'
        '    GOOGLE_API_KEY=AIza...'
    )


def _get(url: str, timeout: int = 60) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode("utf-8", "replace")[:500]}')
    except urllib.error.URLError as e:
        raise RuntimeError(f'Cannot reach {url}: {e.reason}')


def _post(url: str, body: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode("utf-8", "replace")[:500]}')
    except urllib.error.URLError as e:
        raise RuntimeError(f'Cannot reach {url}: {e.reason}')


def list_models(key: str) -> list:
    """Every model this key can see that supports generateContent."""
    out, page = [], ''
    while True:
        url = f'{BASE}/models?key={key}&pageSize=200'
        if page:
            url += f'&pageToken={page}'
        payload = _get(url)
        out.extend(payload.get('models', []))
        page = payload.get('nextPageToken', '')
        if not page:
            break
    return [m for m in out if 'generateContent' in m.get('supportedGenerationMethods', [])]


# Thinking tokens are charged against maxOutputTokens: a 2-line probe spent 446
# on thought and 26 on the answer, so the budget needs real headroom.
def generate(key: str, model: str, prompt: str, max_tokens: int = 2000,
             temperature: float = 0.3) -> str:
    """One generateContent call. Model id may be given with or without 'models/'."""
    name = model if model.startswith('models/') else f'models/{model}'
    payload = _post(f'{BASE}/{name}:generateContent?key={key}', {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'temperature': temperature, 'maxOutputTokens': max_tokens},
    })

    candidates = payload.get('candidates') or []
    if not candidates:
        raise RuntimeError(f'No candidates: {json.dumps(payload)[:400]}')

    # Gemma 4 reasons before answering and returns the scratchpad as a part
    # flagged thought=True, alongside the answer part. Concatenating every part
    # returns the model arguing with itself ("'гие' isn't a word. Let's try...")
    # instead of the two lines asked for.
    parts = candidates[0].get('content', {}).get('parts') or []
    text = ''.join(p.get('text', '') for p in parts if not p.get('thought')).strip()
    if not text:
        # A blocked or truncated response has candidates but no answer part —
        # surface finishReason rather than returning an empty string that looks
        # like output. MAX_TOKENS here means the budget was spent on thinking.
        raise RuntimeError(
            f"Empty text (finishReason={candidates[0].get('finishReason')!r}, "
            f"thoughtsTokenCount={payload.get('usageMetadata', {}).get('thoughtsTokenCount')}): "
            f'{json.dumps(payload)[:300]}'
        )
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='list gemma models and exit')
    parser.add_argument('--all', action='store_true', help="with --list, don't filter to gemma")
    parser.add_argument('--model', default=None, help='model id to probe')
    args = parser.parse_args()

    key = api_key()

    try:
        models = list_models(key)
    except RuntimeError as e:
        sys.exit(f'Could not list models: {e}')

    shown = models if args.all else [m for m in models if 'gemma' in m['name'].lower()]
    print(f'{len(shown)} model(s) usable with this key'
          f"{'' if args.all else ' matching gemma'}, of {len(models)} total:")
    for m in sorted(shown, key=lambda m: m['name']):
        limit = m.get('inputTokenLimit', '?')
        print(f"  {m['name']:<45} in={limit} out={m.get('outputTokenLimit', '?')}")

    if args.list:
        sys.exit(0)

    target = args.model
    if target is None:
        # Prefer the largest Gemma 4 the key can actually see.
        preferred = [m['name'] for m in shown if 'gemma-4-31b' in m['name']]
        target = (preferred or [m['name'] for m in shown] or [None])[0]
    if target is None:
        sys.exit('\nNo gemma model available to this key — rerun with --list --all to see what is.')

    print(f'\nProbing {target} ...')
    try:
        print(generate(key, target, PROBE_PROMPT))
    except RuntimeError as e:
        sys.exit(f'FAILED: {e}')
