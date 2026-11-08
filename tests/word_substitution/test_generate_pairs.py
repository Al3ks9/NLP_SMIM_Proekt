"""Word substitution over another system's pairs, in compare_generations' format."""

import csv

import pytest

from word_substitution import generate_pairs as gp

COLUMNS = ['example_id', 'source_poem_id', 'source_author', 'source_title',
           'target_author', 'split', 'sample_index', 'generated_poem']

POEMS = [
    {'poem_id': '1', 'author': 'Кочо Рацин', 'song_title': 'Балада', 'song_text': 'прв стих'},
    {'poem_id': '2', 'author': 'Ацо Шопов', 'song_title': 'Песна', 'song_text': 'втор стих'},
]


def _write_pairs(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['source_poem_id', 'target_author', 'generated_poem'])
        w.writeheader()
        for pid, target in rows:
            w.writerow({'source_poem_id': pid, 'target_author': target, 'generated_poem': '…'})


def test_read_pairs_keeps_order_and_dedups(tmp_path):
    src = tmp_path / 'other_test.csv'
    _write_pairs(src, [('1', 'A'), ('1', 'A'), ('2', 'B')])
    assert gp.read_pairs(src) == [('1', 'A'), ('2', 'B')]


def test_generate_passes_source_author_and_writes_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, 'load_stripped_songs', lambda: POEMS)
    calls = []

    def fake_transfer(text, target_author, source_author=None):
        calls.append((text, target_author, source_author))
        return 'X'

    out = tmp_path / 'word-substitution_test.csv'
    n = gp.generate([('1', 'Блаже Конески'), ('2', 'Блаже Конески')], out, transfer=fake_transfer)

    assert n == 2
    assert calls == [('прв стих', 'Блаже Конески', 'Кочо Рацин'),
                     ('втор стих', 'Блаже Конески', 'Ацо Шопов')]
    with open(out, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == COLUMNS
    assert [r['generated_poem'] for r in rows] == ['X', 'X']
    assert rows[0]['split'] == 'test' and rows[0]['sample_index'] == '0'


def test_generate_resumes_without_redoing_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, 'load_stripped_songs', lambda: POEMS)
    out = tmp_path / 'ws.csv'
    gp.generate([('1', 'Блаже Конески')], out, transfer=lambda t, a, source_author=None: 'X')
    calls = []
    n = gp.generate([('1', 'Блаже Конески'), ('2', 'Блаже Конески')], out,
                    transfer=lambda t, a, source_author=None: calls.append(t) or 'Y')
    assert n == 1 and calls == ['втор стих']
    with open(out, encoding='utf-8') as f:
        assert len(list(csv.DictReader(f))) == 2


def test_default_transfer_loads_resources_first(tmp_path, monkeypatch):
    from word_substitution import style_transfer as st
    monkeypatch.setattr(gp, 'load_stripped_songs', lambda: POEMS)
    loaded = []
    monkeypatch.setattr(st, 'load_resources', lambda: loaded.append(True))
    monkeypatch.setattr(st, 'transfer_style',
                        lambda t, a, source_author=None: 'Z' if loaded else pytest.fail('not loaded'))
    gp.generate([('1', 'Блаже Конески')], tmp_path / 'ws.csv')
    assert loaded == [True]
