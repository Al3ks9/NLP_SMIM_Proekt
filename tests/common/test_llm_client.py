"""The shared Ollama/OpenRouter call wrapper — dispatch only, no live network calls."""

import io
import json

import pytest

import llm_client as lc


def test_call_dispatches_to_the_named_backend(monkeypatch):
    seen = {}

    def fake_backend(model, prompt, **kw):
        seen['model'] = model
        seen['prompt'] = prompt
        seen['kw'] = kw
        return 'fake response'

    monkeypatch.setitem(lc.BACKENDS, 'fake', fake_backend)

    result = lc.call('some-model', 'some prompt', backend='fake', temperature=0.7)

    assert result == 'fake response'
    assert seen == {'model': 'some-model', 'prompt': 'some prompt', 'kw': {'temperature': 0.7}}


def test_call_rejects_an_unknown_backend():
    with pytest.raises(KeyError):
        lc.call('some-model', 'some prompt', backend='not-a-real-backend')


# ── google backend (Gemini API, direct) ───────────────────────────────────────
#
# Added because every OpenRouter :free Gemma route returns 429 from the shared
# upstream pool. Gemma 4 on this API always reasons before answering — thinking
# cannot be disabled (thinkingBudget returns 400 "not supported for this
# model") — so the scratchpad has to be filtered out of the response.

def _google_response(parts, finish='STOP', usage=None):
    return {'candidates': [{'content': {'parts': parts}, 'finishReason': finish}],
            'usageMetadata': usage or {}}


def test_call_google_returns_the_answer_part(monkeypatch):
    monkeypatch.setattr(lc, '_google_api_key', lambda: 'k')
    monkeypatch.setattr(lc, '_post', lambda *a, **kw: _google_response(
        [{'text': 'одговор'}]))
    assert lc.call_google('gemma-4-31b-it', 'prompt') == 'одговор'


def test_call_google_drops_the_thinking_part(monkeypatch):
    # A 2-line probe spent 446 tokens on thought and 26 on the answer; without
    # this filter the caller gets the model arguing with itself.
    monkeypatch.setattr(lc, '_google_api_key', lambda: 'k')
    monkeypatch.setattr(lc, '_post', lambda *a, **kw: _google_response([
        {'text': "'гие' isn't a word. Let's try 'се крие'.", 'thought': True},
        {'text': 'Под плачечка врба војникот спие,'},
    ]))
    assert lc.call_google('gemma-4-31b-it', 'prompt') == 'Под плачечка врба војникот спие,'


def test_call_google_raises_when_thinking_consumed_the_whole_budget(monkeypatch):
    # finishReason MAX_TOKENS with no answer part: returning '' here would look
    # like the model declined rather than like a budget that was set too low.
    monkeypatch.setattr(lc, '_google_api_key', lambda: 'k')
    monkeypatch.setattr(lc, '_post', lambda *a, **kw: _google_response(
        [{'text': 'thinking...', 'thought': True}],
        finish='MAX_TOKENS', usage={'thoughtsTokenCount': 446}))
    with pytest.raises(RuntimeError, match='MAX_TOKENS'):
        lc.call_google('gemma-4-31b-it', 'prompt')


def test_call_google_sends_no_thinking_config(monkeypatch):
    # thinkingBudget is a 400 on Gemma 4 — sending it would break every call.
    monkeypatch.setattr(lc, '_google_api_key', lambda: 'k')
    sent = {}

    def fake_post(url, body, headers, timeout):
        sent['body'] = body
        return _google_response([{'text': 'ok'}])

    monkeypatch.setattr(lc, '_post', fake_post)
    lc.call_google('gemma-4-31b-it', 'prompt')
    assert 'thinkingConfig' not in sent['body']['generationConfig']


def test_call_google_budgets_far_more_tokens_than_the_other_backends():
    # Thinking is charged against maxOutputTokens, so the 700 the other
    # backends use would leave nothing for the poem.
    import inspect
    default = inspect.signature(lc.call_google).parameters['max_tokens'].default
    assert default >= 2000


def test_google_is_a_registered_backend():
    assert lc.BACKENDS['google'] is lc.call_google


# ── google backend: thinking-length variance ──────────────────────────────────
#
# Thinking length swings run to run for the same prompt: one style-transfer
# prompt spent 3231 thought tokens, a second 15997 (blowing a 16000 budget with
# no answer at all), and the same prompt at a 32768 budget then spent 2075. A
# budget alone cannot cover that spread, so a starved call is retried.

def _max_tokens_response():
    return _google_response([{'text': 'thinking...', 'thought': True}],
                            finish='MAX_TOKENS', usage={'thoughtsTokenCount': 15997})


def test_call_google_retries_when_thinking_starved_the_answer(monkeypatch):
    monkeypatch.setattr(lc, '_google_api_key', lambda: 'k')
    calls = []

    def fake_post(url, body, headers, timeout):
        calls.append(body['generationConfig']['maxOutputTokens'])
        if len(calls) == 1:
            return _max_tokens_response()
        return _google_response([{'text': 'песна'}])

    monkeypatch.setattr(lc, '_post', fake_post)
    assert lc.call_google('gemma-4-31b-it', 'prompt') == 'песна'
    assert len(calls) == 2


def test_call_google_gives_up_after_the_retry_budget(monkeypatch):
    monkeypatch.setattr(lc, '_google_api_key', lambda: 'k')
    calls = []

    def fake_post(url, body, headers, timeout):
        calls.append(1)
        return _max_tokens_response()

    monkeypatch.setattr(lc, '_post', fake_post)
    with pytest.raises(RuntimeError, match='MAX_TOKENS'):
        lc.call_google('gemma-4-31b-it', 'prompt', attempts=3)
    assert len(calls) == 3


def test_call_google_does_not_retry_a_successful_call(monkeypatch):
    monkeypatch.setattr(lc, '_google_api_key', lambda: 'k')
    calls = []

    def fake_post(url, body, headers, timeout):
        calls.append(1)
        return _google_response([{'text': 'песна'}])

    monkeypatch.setattr(lc, '_post', fake_post)
    lc.call_google('gemma-4-31b-it', 'prompt')
    assert len(calls) == 1


def test_call_google_does_not_retry_a_genuinely_empty_candidate(monkeypatch):
    # No candidates at all is a different failure (blocked/malformed) — retrying
    # it just burns quota.
    monkeypatch.setattr(lc, '_google_api_key', lambda: 'k')
    calls = []

    def fake_post(url, body, headers, timeout):
        calls.append(1)
        return {'candidates': []}

    monkeypatch.setattr(lc, '_post', fake_post)
    with pytest.raises(RuntimeError, match='No candidates'):
        lc.call_google('gemma-4-31b-it', 'prompt')
    assert len(calls) == 1


def test_call_google_budget_matches_the_model_output_ceiling():
    import inspect
    assert inspect.signature(lc.call_google).parameters['max_tokens'].default == 32768


# ── _post: retry on transient HTTP errors ──────────────────────────────────────
#
# call_google was hitting Gemini's own 500s/503s ("high demand") with no retry
# at all -- a single transient blip failed the whole pair in generate_synthetic.py.
# Retries live in _post (shared by all three backends) rather than per-backend.

def _http_error(code, body=b'error'):
    return lc.urllib.error.HTTPError('http://x', code, 'msg', {}, io.BytesIO(body))


def _ok_response(payload):
    return io.BytesIO(json.dumps(payload).encode('utf-8'))


def test_post_retries_a_503_then_succeeds(monkeypatch):
    monkeypatch.setattr(lc.time, 'sleep', lambda s: None)
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(503)
        return _ok_response({'ok': True})

    monkeypatch.setattr(lc.urllib.request, 'urlopen', fake_urlopen)
    result = lc._post('http://x', {}, {}, timeout=10)
    assert result == {'ok': True}
    assert len(calls) == 2


def test_post_does_not_retry_a_client_error(monkeypatch):
    # 400/403 mean a bad request or bad key -- every retry would fail the same
    # way, so this must not sleep or call urlopen a second time.
    monkeypatch.setattr(lc.time, 'sleep', lambda s: pytest.fail('should not retry'))
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _http_error(400)

    monkeypatch.setattr(lc.urllib.request, 'urlopen', fake_urlopen)
    with pytest.raises(RuntimeError, match='HTTP 400'):
        lc._post('http://x', {}, {}, timeout=10)
    assert len(calls) == 1


def test_post_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(lc.time, 'sleep', lambda s: None)
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _http_error(503)

    monkeypatch.setattr(lc.urllib.request, 'urlopen', fake_urlopen)
    with pytest.raises(RuntimeError, match='HTTP 503'):
        lc._post('http://x', {}, {}, timeout=10, max_retries=2)
    assert len(calls) == 3  # initial attempt + 2 retries


def test_post_retries_a_429(monkeypatch):
    monkeypatch.setattr(lc.time, 'sleep', lambda s: None)
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(429)
        return _ok_response({'ok': True})

    monkeypatch.setattr(lc.urllib.request, 'urlopen', fake_urlopen)
    assert lc._post('http://x', {}, {}, timeout=10) == {'ok': True}


def test_post_backoff_grows_exponentially(monkeypatch):
    # Jitter is a random.uniform(0, BASE_DELAY) add-on -- pin it to 0 so the
    # sleep sequence is checkable exactly: BASE_DELAY * 2**attempt.
    monkeypatch.setattr(lc.random, 'uniform', lambda a, b: 0)
    delays = []
    monkeypatch.setattr(lc.time, 'sleep', delays.append)
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _http_error(503)

    monkeypatch.setattr(lc.urllib.request, 'urlopen', fake_urlopen)
    with pytest.raises(RuntimeError):
        lc._post('http://x', {}, {}, timeout=10, max_retries=3)
    assert delays == [lc.BASE_DELAY * 1, lc.BASE_DELAY * 2, lc.BASE_DELAY * 4]
