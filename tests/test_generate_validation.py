"""Qwen SFT pipeline, stage 6: validation generation. Model/tokenizer/LoRA
loading is monkeypatched throughout -- no GPU or model download needed to
exercise the orchestration and metadata logic."""

import csv
import json

import torch

import generate_validation as gv
import llm_style_transfer as lst


def _poem(poem_id, author, title='Наслов', text='текст на песна'):
    return {'poem_id': poem_id, 'author': author, 'song_title': title, 'song_text': text}


# ── _infer_base_model ────────────────────────────────────────────────────────────

def test_infer_base_model_reads_adapter_config(tmp_path):
    (tmp_path / 'adapter_config.json').write_text(
        json.dumps({'base_model_name_or_path': 'Qwen/Qwen3-4B'}), encoding='utf-8')
    assert gv._infer_base_model(tmp_path) == 'Qwen/Qwen3-4B'


def test_infer_base_model_falls_back_when_config_missing(tmp_path):
    assert gv._infer_base_model(tmp_path) == gv.DEFAULT_MODEL_NAME


# ── generate_one ─────────────────────────────────────────────────────────────────

class _FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return 'PROMPT'

    def __call__(self, text, return_tensors):
        assert text == 'PROMPT'
        return _FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is True
        return ' нова песна '


class FakeModel:
    device = 'cpu'

    def generate(self, **kw):
        assert kw['input_ids'].shape == (1, 3)
        assert 'generator' not in kw  # generate() takes no generator= kwarg
        # 3 prompt tokens + 2 "generated" tokens
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_generate_one_returns_only_the_newly_generated_text():
    text = gv.generate_one(FakeModel(), FakeTokenizer(), 'извор', 'Автор')
    assert text == 'нова песна'


# ── run_validation orchestration ─────────────────────────────────────────────────

def test_run_validation_writes_expected_columns_and_uses_the_given_split(monkeypatch, tmp_path):
    monkeypatch.setattr(gv, 'select_target_authors', lambda n, explicit=None: ['Ц1'])
    monkeypatch.setattr(gv, 'sample_source_poems',
                        lambda n, seed, split: [_poem('0', 'А'), _poem('1', 'Б')])
    monkeypatch.setattr(gv, 'load_adapter_model',
                        lambda adapter_dir, base_model=None: (FakeModel(), FakeTokenizer()))
    monkeypatch.setattr(gv, 'generate_one', lambda *a, **kw: 'генерирана песна')
    monkeypatch.setattr(lst, 'structural_targets', lambda author: {
        'avg_lines_per_poem': 2, 'avg_tokens_per_line': 2,
        'avg_stanzas_per_poem': 1, 'avg_lines_per_stanza': 2})

    out_path = tmp_path / 'out.csv'
    result_path = gv.run_validation(
        adapter_dir=tmp_path / 'adapter', split='val', num_source_poems=2,
        num_target_authors=1, output_path=out_path)

    assert result_path == out_path
    with open(out_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(r['split'] == 'val' for r in rows)
    assert all(r['target_author'] == 'Ц1' for r in rows)
    assert all(r['generated_poem'] == 'генерирана песна' for r in rows)
    assert {'delta_lines', 'delta_stanzas', 'delta_tokens_per_line',
           'delta_lines_per_stanza'} <= set(rows[0].keys())


def test_run_validation_never_touches_train_split_unless_asked(monkeypatch, tmp_path):
    seen_splits = []
    monkeypatch.setattr(gv, 'select_target_authors', lambda n, explicit=None: ['Ц1'])

    def fake_sample(n, seed, split):
        seen_splits.append(split)
        return [_poem('0', 'А')]

    monkeypatch.setattr(gv, 'sample_source_poems', fake_sample)
    monkeypatch.setattr(gv, 'load_adapter_model',
                        lambda adapter_dir, base_model=None: (FakeModel(), FakeTokenizer()))
    monkeypatch.setattr(gv, 'generate_one', lambda *a, **kw: 'п')
    monkeypatch.setattr(lst, 'structural_targets', lambda author: {
        'avg_lines_per_poem': 1, 'avg_tokens_per_line': 1,
        'avg_stanzas_per_poem': 1, 'avg_lines_per_stanza': 1})

    gv.run_validation(adapter_dir=tmp_path / 'adapter', split='val', num_source_poems=1,
                      num_target_authors=1, output_path=tmp_path / 'out.csv')
    assert seen_splits == ['val']
