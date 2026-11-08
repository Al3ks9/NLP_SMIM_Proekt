"""Human-evaluation items: which pairs, which panel poems, which control poem."""

import pytest

from students.shared import human_eval as he


def _poem(pid, author, n_lines, text=None):
    return {'poem_id': pid, 'author': author, 'song_title': f't{pid}',
            'song_text': text or '\n'.join(f'стих {i}' for i in range(n_lines))}


def _rows(pairs, poem='песна'):
    return [{'source_poem_id': p, 'target_author': t, 'generated_poem': f'{poem} {p} {t}'}
            for p, t in pairs]


ALL = [(str(p), a) for a in ('A', 'B') for p in range(1, 5)]


def test_select_pairs_only_common_and_per_author():
    rows = {'s1': _rows(ALL), 's2': _rows(ALL), 's3': _rows([k for k in ALL if k != ('1', 'A')])}
    chosen = he.select_pairs(rows, per_author=2, seed=7)
    assert ('1', 'A') not in chosen
    assert sorted(t for _, t in chosen) == ['A', 'A', 'B', 'B']
    assert he.select_pairs(rows, per_author=2, seed=7) == chosen


def test_select_pairs_raises_when_author_short():
    rows = {'s1': _rows(ALL), 's2': _rows([k for k in ALL if k[1] == 'A'] + [('1', 'B')])}
    with pytest.raises(ValueError, match='B'):
        he.select_pairs(rows, per_author=2)


def test_panel_prefers_8_to_30_lines(monkeypatch):
    train = [_poem('1', 'A', 3), _poem('2', 'A', 10), _poem('3', 'A', 20),
             _poem('4', 'A', 25), _poem('5', 'A', 60), _poem('6', 'B', 10)]
    monkeypatch.setattr(he, 'poems_in_split', lambda split: train)
    panel = he.panel_poems('A', k=3)
    assert sorted(p['poem_id'] for p in panel) == ['2', '3', '4']


def test_panel_falls_back_to_any_length(monkeypatch):
    train = [_poem('1', 'A', 3), _poem('2', 'A', 10), _poem('5', 'A', 60)]
    monkeypatch.setattr(he, 'poems_in_split', lambda split: train)
    assert sorted(p['poem_id'] for p in he.panel_poems('A', k=3)) == ['1', '2', '5']


def test_control_excludes_panel_and_picks_max_f1(monkeypatch):
    train = [_poem('1', 'A', 10, 'x1'), _poem('2', 'A', 10, 'x2'), _poem('3', 'A', 10, 'x3')]
    monkeypatch.setattr(he, 'poems_in_split', lambda split: train)
    score = {'x1': 0.9, 'x2': 0.7, 'x3': 0.1}
    chosen = he.control_poem('извор', 'A', exclude_ids={'1'}, f1=lambda src, cand: score[cand])
    assert chosen['poem_id'] == '2'


def test_build_items_counts_and_ids(monkeypatch):
    corpus = [_poem(str(p), 'S', 5) for p in range(1, 5)]
    train = [_poem(str(p), a, 10) for a, ps in (('A', range(10, 16)), ('B', range(20, 26))) for p in ps]
    monkeypatch.setattr(he, 'load_stripped_songs', lambda: corpus + train)
    monkeypatch.setattr(he, 'poems_in_split', lambda split: train)
    rows = {'teacher': _rows(ALL, 'T'), 'qwen-sft': _rows(ALL, 'Q')}

    items, key, panels = he.build_items(rows, ['teacher', 'qwen-sft'], seed=1,
                                        f1=lambda src, cand: len(cand))

    assert len(items) == 2 * 2 * (2 + 1)                     # authors x pairs x (systems + control)
    ids = [it['item_id'] for it in items]
    assert len(set(ids)) == len(ids) and all(len(i) == 10 for i in ids)
    assert set(key['items']) == set(ids)
    assert key['page_version'] == he.page_version(items)
    assert {v['system'] for v in key['items'].values()} == {'teacher', 'qwen-sft', 'real-poem'}
    assert all('system' not in it for it in items)
    assert set(panels) == {'A', 'B'} and all(len(v) == 3 for v in panels.values())
    panel_titles = {p['title'] for v in panels.values() for p in v}
    control_ids = [v['control_poem_id'] for v in key['items'].values() if v['system'] == 'real-poem']
    assert len(control_ids) == 4
    assert not any(f't{pid}' in panel_titles for pid in control_ids)


# --- summarising ratings ------------------------------------------------------

import json
import math

SCALES = ['macedonian', 'coherent', 'poetic', 'content', 'author', 'usable']


def test_alpha_perfect_agreement():
    assert he.krippendorff_alpha_interval([[1, 1], [2, 2], [3, 3]]) == 1.0


def test_alpha_known_value():
    # values 1,2,2,1: Do = 1, De = 8/12  ->  alpha = 1 - 1.5 = -0.5
    assert abs(he.krippendorff_alpha_interval([[1, 2], [2, 1]]) + 0.5) < 1e-9


def test_alpha_needs_two_pairable_units():
    assert math.isnan(he.krippendorff_alpha_interval([[3, 3], [4]]))


def _ratings_file(path, rater, version, scores_by_item, started='t0'):
    path.write_text(json.dumps({
        'rater': rater, 'page_version': version, 'started': started, 'finished': 't1',
        'ratings': [{'item_id': i, 'scores': {s: v for s in SCALES}, 'comment': '', 'seconds': 5}
                    for i, v in scores_by_item.items()]}), encoding='utf-8')


def _key(path):
    path.write_text(json.dumps({'page_version': 'v1', 'items': {
        'a': {'system': 'teacher', 'source_poem_id': '1', 'target_author': 'A'},
        'b': {'system': 'teacher', 'source_poem_id': '2', 'target_author': 'A'},
        'c': {'system': 'real-poem', 'source_poem_id': '1', 'target_author': 'A'},
    }}), encoding='utf-8')
    return path


def test_summarize_means_and_control_has_no_content(tmp_path):
    key = _key(tmp_path / 'items_key.json')
    _ratings_file(tmp_path / 'ratings_x.json', 'Алекс', 'v1', {'a': 4, 'b': 2, 'c': 5})
    rows = he.summarize(tmp_path, key)
    teacher = {r['scale']: r for r in rows if r['system'] == 'teacher'}
    assert teacher['poetic']['mean'] == 3.0 and teacher['poetic']['n'] == 2
    assert teacher['poetic']['ci_low'] <= 3.0 <= teacher['poetic']['ci_high']
    control_scales = {r['scale'] for r in rows if r['system'] == 'real-poem'}
    assert 'content' not in control_scales and 'author' in control_scales
    assert not any(r['system'] == 'ALL' for r in rows)          # one rater: no alpha


def test_summarize_rejects_other_page_version(tmp_path):
    key = _key(tmp_path / 'items_key.json')
    _ratings_file(tmp_path / 'ratings_old.json', 'Алекс', 'v0', {'a': 4})
    with pytest.raises(ValueError, match='ratings_old.json'):
        he.summarize(tmp_path, key)


def test_summarize_same_name_two_files_are_two_raters(tmp_path):
    key = _key(tmp_path / 'items_key.json')
    _ratings_file(tmp_path / 'ratings_1.json', 'Алекс', 'v1', {'a': 4, 'b': 2, 'c': 5}, started='t0')
    _ratings_file(tmp_path / 'ratings_2.json', 'Алекс', 'v1', {'a': 4, 'b': 2, 'c': 5}, started='t9')
    rows = he.summarize(tmp_path, key)
    alpha = {r['scale']: r['alpha'] for r in rows if r['system'] == 'ALL'}
    assert alpha['poetic'] == 1.0


# --- final-review fixes -------------------------------------------------------

def _two_pair_setup(monkeypatch, train_texts=None, gen='песна'):
    corpus = [_poem(str(p), 'S', 5) for p in range(1, 5)]
    train = [_poem(str(p), 'A', 10, (train_texts or {}).get(str(p))) for p in range(10, 17)]
    monkeypatch.setattr(he, 'load_stripped_songs', lambda: corpus + train)
    monkeypatch.setattr(he, 'poems_in_split', lambda split: train)
    pairs = [(str(p), 'A') for p in range(1, 5)]
    return {'teacher': _rows(pairs, gen)}


def test_build_items_normalises_whitespace_of_every_shown_poem(monkeypatch):
    texts = {str(p): '\n\nпрв стих\n\n\n\nвтор стих\n' for p in range(10, 17)}
    rows = _two_pair_setup(monkeypatch, texts)
    items, key, _ = he.build_items(rows, ['teacher'], seed=1, f1=lambda a, b: 0.0)
    controls = [it for it in items if key['items'][it['item_id']]['system'] == 'real-poem']
    assert controls and all(c['poem'] == 'прв стих\n\nвтор стих' for c in controls)
    assert all(it['poem'] == it['poem'].strip() and '\n\n\n' not in it['poem'] for it in items)
    assert all(it['source']['text'] == it['source']['text'].strip() for it in items)


def test_controls_are_distinct_within_a_target(monkeypatch):
    rows = _two_pair_setup(monkeypatch)
    items, key, panels = he.build_items(rows, ['teacher'], seed=1,
                                        f1=lambda a, b: 0.0)          # every candidate ties
    ids = [v['control_poem_id'] for v in key['items'].values() if v['system'] == 'real-poem']
    assert len(ids) == 2 and len(set(ids)) == 2


def test_page_version_changes_with_poem_text(monkeypatch):
    v1 = he.build_items(_two_pair_setup(monkeypatch, gen='прва'), ['teacher'], seed=1,
                        f1=lambda a, b: 0.0)[1]['page_version']
    v2 = he.build_items(_two_pair_setup(monkeypatch, gen='друга'), ['teacher'], seed=1,
                        f1=lambda a, b: 0.0)[1]['page_version']
    assert v1 != v2


def test_summarize_counts_a_reexport_once(tmp_path):
    key = _key(tmp_path / 'items_key.json')
    _ratings_file(tmp_path / 'ratings_Алекс.json', 'Алекс', 'v1', {'a': 4})
    _ratings_file(tmp_path / 'ratings_Алекс (1).json', 'Алекс', 'v1', {'a': 4, 'b': 2, 'c': 5})
    rows = he.summarize(tmp_path, key)
    assert not any(r['system'] == 'ALL' for r in rows)                 # one rater: no alpha
    teacher = {r['scale']: r for r in rows if r['system'] == 'teacher'}
    assert teacher['poetic']['n'] == 2 and teacher['poetic']['mean'] == 3.0   # the fuller file


# --- reveal after rating ------------------------------------------------------

def test_reveal_names_the_system_or_the_real_poem(monkeypatch):
    rows = _two_pair_setup(monkeypatch)
    items, key, _ = he.build_items(rows, ['teacher'], seed=1, f1=lambda a, b: 0.0, reveal=True)
    for it in items:
        entry = key['items'][it['item_id']]
        if entry['system'] == 'real-poem':
            assert it['reveal'] == f"Оригинална песна — A, „t{entry['control_poem_id']}“"
        else:
            assert it['reveal'] == 'Генерирана песна — teacher'


def test_reveal_is_off_by_default_and_keeps_the_page_version(monkeypatch):
    plain, key, _ = he.build_items(_two_pair_setup(monkeypatch), ['teacher'], seed=1,
                                   f1=lambda a, b: 0.0)
    shown, key2, _ = he.build_items(_two_pair_setup(monkeypatch), ['teacher'], seed=1,
                                    f1=lambda a, b: 0.0, reveal=True)
    assert not any('reveal' in it for it in plain)
    assert key['page_version'] == key2['page_version']      # saved progress carries over
