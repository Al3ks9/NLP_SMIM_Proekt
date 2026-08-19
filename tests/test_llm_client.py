"""The shared Ollama/OpenRouter call wrapper — dispatch only, no live network calls."""

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
