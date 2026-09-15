"""Qwen SFT pipeline, stage 2: synthetic style-transfer dataset generation.
run_transfer is monkeypatched -- no live LLM calls, matching
test_llm_style_transfer.py's approach."""

import csv

import pytest

import generate_synthetic as gs


def _poem(poem_id, author, title='Наслов'):
    return {'poem_id': poem_id, 'author': author, 'song_title': title,
           'song_text': 'текст'}


# ── sample_source_poems ──────────────────────────────────────────────────────────

def test_sample_source_poems_is_reproducible_given_the_same_seed(monkeypatch):
    pool = [_poem(str(i), 'А') for i in range(20)]
    monkeypatch.setattr(gs, 'poems_in_split', lambda split: pool)
    first = gs.sample_source_poems(5, seed=7)
    second = gs.sample_source_poems(5, seed=7)
    assert first == second


def test_sample_source_poems_raises_when_more_requested_than_available(monkeypatch):
    monkeypatch.setattr(gs, 'poems_in_split', lambda split: [_poem('0', 'А')])
    with pytest.raises(ValueError):
        gs.sample_source_poems(5)


# ── build_pairs ──────────────────────────────────────────────────────────────────

def test_build_pairs_crosses_every_poem_with_every_target_author():
    poems = [_poem('0', 'А'), _poem('1', 'Б')]
    pairs = gs.build_pairs(poems, ['Ц1', 'Ц2'], samples_per_pair=1)
    assert len(pairs) == 4


def test_build_pairs_skips_self_transfer():
    poems = [_poem('0', 'Ц1')]
    pairs = gs.build_pairs(poems, ['Ц1', 'Ц2'], samples_per_pair=1)
    assert len(pairs) == 1
    assert pairs[0][1] == 'Ц2'


def test_build_pairs_respects_samples_per_pair():
    poems = [_poem('0', 'А')]
    pairs = gs.build_pairs(poems, ['Ц'], samples_per_pair=3)
    assert [p[2] for p in pairs] == [0, 1, 2]


# ── load_done_keys ────────────────────────────────────────────────────────────────

def test_load_done_keys_returns_empty_set_for_a_missing_file(tmp_path):
    assert gs.load_done_keys(tmp_path / 'nope.csv') == set()


def test_load_done_keys_reads_back_existing_rows(tmp_path):
    path = tmp_path / 'synthetic_dataset.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=gs.FIELDNAMES)
        writer.writeheader()
        writer.writerow({**{k: '' for k in gs.FIELDNAMES},
                        'source_poem_id': '3', 'target_author': 'Ц', 'sample_index': '0'})
    assert gs.load_done_keys(path) == {('3', 'Ц', 0)}


# ── generate() orchestration ──────────────────────────────────────────────────────

def _fake_run_transfer(source_author, source_title, target_author, model=None,
                       backend=None, source_poem_id=None, **kw):
    return {
        'source_poem_id': source_poem_id, 'source_author': source_author,
        'source_title': source_title, 'target_author': target_author,
        'generated_poem': f'песна за {target_author}', 'model': model, 'backend': backend,
        'structural_fit': {'delta_lines': 0, 'delta_stanzas': 0,
                           'delta_tokens_per_line': 0, 'delta_lines_per_stanza': 0},
        'timestamp': '2026-01-01T00:00:00+00:00', 'log_path': 'x.json',
    }


def test_generate_writes_one_row_per_pair(monkeypatch, tmp_path):
    monkeypatch.setattr(gs, 'select_target_authors', lambda n, explicit=None: ['Ц1', 'Ц2'])
    monkeypatch.setattr(gs, 'sample_source_poems', lambda n, seed: [_poem('0', 'А'), _poem('1', 'Б')])
    monkeypatch.setattr(gs.lst, 'run_transfer', _fake_run_transfer)

    result = gs.generate(num_target_authors=2, poems_per_author=2,
                        output_path=tmp_path / 'out.csv', errors_path=tmp_path / 'err.csv')
    assert result['generated'] == 4
    assert result['failed'] == 0

    with open(tmp_path / 'out.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert all(r['split'] == 'train' for r in rows)
    assert {r['target_author'] for r in rows} == {'Ц1', 'Ц2'}


def test_generate_is_resumable_and_skips_already_done_pairs(monkeypatch, tmp_path):
    monkeypatch.setattr(gs, 'select_target_authors', lambda n, explicit=None: ['Ц1'])
    monkeypatch.setattr(gs, 'sample_source_poems', lambda n, seed: [_poem('0', 'А'), _poem('1', 'Б')])
    calls = []

    def counting_run_transfer(*a, **kw):
        calls.append(kw.get('source_poem_id'))
        return _fake_run_transfer(*a, **kw)

    monkeypatch.setattr(gs.lst, 'run_transfer', counting_run_transfer)
    out_path, err_path = tmp_path / 'out.csv', tmp_path / 'err.csv'

    gs.generate(num_target_authors=1, poems_per_author=2, output_path=out_path, errors_path=err_path)
    assert len(calls) == 2

    calls.clear()
    result = gs.generate(num_target_authors=1, poems_per_author=2, output_path=out_path, errors_path=err_path)
    assert calls == []
    assert result['generated'] == 0


def test_generate_logs_failures_without_aborting_the_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(gs, 'select_target_authors', lambda n, explicit=None: ['Ц1'])
    monkeypatch.setattr(gs, 'sample_source_poems', lambda n, seed: [_poem('0', 'А'), _poem('1', 'Б')])

    def flaky_run_transfer(source_author, source_title, target_author, source_poem_id=None, **kw):
        if source_poem_id == '0':
            raise RuntimeError('quota exceeded')
        return _fake_run_transfer(source_author, source_title, target_author,
                                  source_poem_id=source_poem_id, **kw)

    monkeypatch.setattr(gs.lst, 'run_transfer', flaky_run_transfer)
    out_path, err_path = tmp_path / 'out.csv', tmp_path / 'err.csv'
    result = gs.generate(num_target_authors=1, poems_per_author=2, output_path=out_path, errors_path=err_path)

    assert result['generated'] == 1
    assert result['failed'] == 1
    with open(err_path, encoding='utf-8') as f:
        errors = list(csv.DictReader(f))
    assert len(errors) == 1
    assert errors[0]['source_poem_id'] == '0'
