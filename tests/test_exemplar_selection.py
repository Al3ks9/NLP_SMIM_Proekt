"""Exemplar line selector (step 4): line embedding + k-means representative
selection are pure and tested on synthetic vectors; select_exemplars() is
tested end-to-end against the real corpus, like test_candidate_selection.py."""

import numpy as np
import pytest

import exemplar_selection as es

RACIN = 'Кочо Рацин'
KONESKI = 'Блаже Конески'


# ── line_embedding: centroid of content-word vectors ──────────────────────────

def test_line_embedding_is_the_mean_of_its_lemma_vectors():
    embeddings = {
        'сокол': np.array([1.0, 0.0]),
        'мрачен': np.array([0.0, 1.0]),
    }
    vec = es.line_embedding(['сокол', 'мрачен'], embeddings)
    assert vec == pytest.approx([0.5, 0.5])


def test_line_embedding_skips_lemmas_with_no_vector():
    embeddings = {'сокол': np.array([1.0, 0.0])}
    vec = es.line_embedding(['сокол', 'непознат_збор'], embeddings)
    assert vec == pytest.approx([1.0, 0.0])


def test_line_embedding_returns_none_when_nothing_is_embeddable():
    assert es.line_embedding(['непознат_збор'], {}) is None


# ── cluster_and_select: k-means + closest-to-centroid representatives ─────────

def test_cluster_and_select_picks_lines_from_distinct_clusters():
    # Two obvious clusters around (0,0) and (10,10); each has 3 candidate lines.
    candidates = [
        {'line': f'line{i}', 'vector': np.array(v)}
        for i, v in enumerate([
            [0.0, 0.1], [0.1, 0.0], [0.0, 0.0],
            [10.0, 10.1], [10.1, 10.0], [10.0, 10.0],
        ])
    ]
    chosen = es.cluster_and_select(candidates, k=2, per_cluster=1)
    assert len(chosen) == 2
    clusters = {c['cluster'] for c in chosen}
    assert clusters == {0, 1}


def test_cluster_and_select_respects_per_cluster_cap():
    candidates = [
        {'line': f'line{i}', 'vector': np.array([float(i), 0.0])}
        for i in range(6)
    ]
    chosen = es.cluster_and_select(candidates, k=1, per_cluster=2)
    assert len(chosen) == 2


def test_cluster_and_select_reduces_k_when_fewer_candidates_than_k():
    candidates = [
        {'line': 'only one', 'vector': np.array([1.0, 1.0])},
    ]
    chosen = es.cluster_and_select(candidates, k=5, per_cluster=2)
    assert len(chosen) == 1


def test_cluster_and_select_returns_empty_for_no_candidates():
    assert es.cluster_and_select([], k=5, per_cluster=2) == []


# ── select_exemplars: end-to-end against the real corpus ───────────────────────

def test_select_exemplars_returns_lines_attributed_to_the_target_author():
    exemplars = es.select_exemplars(KONESKI, k=5, per_cluster=2)
    assert exemplars
    assert all(e['author'] == KONESKI for e in exemplars)
    assert all(e['line'].strip() for e in exemplars)


def test_select_exemplars_caps_lines_per_source_poem():
    exemplars = es.select_exemplars(KONESKI, k=5, per_cluster=2, max_lines_per_poem=2)
    from collections import Counter
    counts = Counter(e['poem_id'] for e in exemplars)
    assert all(c == 1 for c in counts.values())


def test_select_exemplars_covers_multiple_poems():
    # per_cluster capped lines from a single poem would defeat the "force
    # multi-poem coverage" point of the 2-line cap.
    exemplars = es.select_exemplars(KONESKI, k=5, per_cluster=2)
    titles = {e['song_title'] for e in exemplars}
    assert len(titles) > 1


# ── qualifying_lines: de-duplication ──────────────────────────────────────────
#
# Duplicate exemplar lines reached the LLM prompt from two directions: a refrain
# repeated inside one poem ate the whole per-poem quota, and the corpus's
# near-duplicate poems (Конески's 'Бура' / 'БУРА') each contributed the same
# line independently. Both showed up in data/llm_transfer_logs.

def _stub_corpus(monkeypatch, rows):
    """qualifying_lines() over synthetic stripped_songs rows, with the pos/lemma
    machinery stubbed out — dedup is a property of the line text alone."""
    monkeypatch.setattr(es, 'load_stripped_songs', lambda: rows)
    monkeypatch.setattr(es, 'load_pos_rows', lambda: [])
    monkeypatch.setattr(es, 'corpus_token_stream', lambda poem_id, pos_rows: [])


def test_qualifying_lines_skips_a_line_repeated_within_one_poem(monkeypatch):
    # Шопов's 'Треба да бидеме подобри' opens with the same line twice; the
    # 2-lines-per-poem cap used to spend both slots on it.
    _stub_corpus(monkeypatch, [
        {'poem_id': '1', 'author': 'А', 'song_title': 'Рефрен',
         'song_text': 'сонот се врти\nсонот се врти\nноќта е долга\n'},
    ])
    lines = [c['line'] for c in es.qualifying_lines('А', {'сонот', 'ноќта'},
                                                    max_lines_per_poem=2)]
    assert lines == ['сонот се врти', 'ноќта е долга']


def test_qualifying_lines_skips_a_line_already_taken_from_another_poem(monkeypatch):
    # 'Бура' (poem 13) and 'БУРА' (poem 566) are the same Конески poem under
    # two title casings — dedup is author-wide, not per-poem.
    _stub_corpus(monkeypatch, [
        {'poem_id': '13', 'author': 'А', 'song_title': 'Бура',
         'song_text': 'ветрот носи глас\n'},
        {'poem_id': '566', 'author': 'А', 'song_title': 'БУРА',
         'song_text': 'ветрот носи глас\n'},
    ])
    got = es.qualifying_lines('А', {'ветрот'}, max_lines_per_poem=2)
    assert [c['line'] for c in got] == ['ветрот носи глас']


def test_qualifying_lines_dedup_ignores_case_and_whitespace(monkeypatch):
    _stub_corpus(monkeypatch, [
        {'poem_id': '1', 'author': 'А', 'song_title': 'П',
         'song_text': 'Ветрот носи глас\n  ВЕТРОТ   носи  глас  \n'},
    ])
    got = es.qualifying_lines('А', {'ветрот'}, max_lines_per_poem=2)
    assert len(got) == 1


def test_qualifying_lines_carries_poem_id(monkeypatch):
    _stub_corpus(monkeypatch, [
        {'poem_id': '42', 'author': 'А', 'song_title': 'П',
         'song_text': 'ветрот носи глас\n'},
    ])
    assert es.qualifying_lines('А', {'ветрот'})[0]['poem_id'] == '42'


# ── cluster_and_select: one line per source poem ──────────────────────────────

def _cand(line, poem_id, vector):
    return {'line': line, 'poem_id': poem_id, 'vector': np.array(vector)}


def test_cluster_and_select_prefers_lines_from_unclaimed_poems():
    # The two closest to the centroid share a poem; the runner-up from a
    # different poem should take the second slot instead.
    candidates = [
        _cand('near a', 'p1', [0.0, 0.0]),
        _cand('near b', 'p1', [0.1, 0.0]),
        _cand('further', 'p2', [0.5, 0.0]),
    ]
    chosen = es.cluster_and_select(candidates, k=1, per_cluster=2)
    assert {c['poem_id'] for c in chosen} == {'p1', 'p2'}
    # centroid is the mean (0.2, 0), so 'near b' is the closest; 'near a' is
    # next but shares p1 with it, so the slot goes to 'further'.
    assert [c['line'] for c in chosen] == ['near b', 'further']


def test_cluster_and_select_returns_fewer_rather_than_two_lines_from_one_poem():
    # A cluster whose remaining members are all from claimed poems gives up the
    # slot. Corpus-wide that costs 7 lines out of 218 and buys a block where
    # every line comes from a different poem.
    candidates = [
        _cand('one', 'p1', [0.0, 0.0]),
        _cand('two', 'p1', [0.1, 0.0]),
    ]
    chosen = es.cluster_and_select(candidates, k=1, per_cluster=2)
    assert [c['line'] for c in chosen] == ['one']


def test_cluster_and_select_lets_the_smallest_cluster_claim_its_poem_first():
    # Шопов's cluster 3 had a single member, from a poem that the 8-member
    # cluster 0 also contained; processing in cluster-id order cost cluster 3
    # its only candidate.
    candidates = [
        _cand('big a', 'p1', [0.0, 0.0]),
        _cand('big b', 'p2', [0.1, 0.0]),
        _cand('big c', 'p3', [0.2, 0.0]),
        _cand('lone', 'p1', [10.0, 10.0]),
    ]
    chosen = es.cluster_and_select(candidates, k=2, per_cluster=1)
    assert 'lone' in [c['line'] for c in chosen]
    assert len({c['poem_id'] for c in chosen}) == len(chosen)


def test_cluster_and_select_tolerates_candidates_without_a_poem_id():
    candidates = [
        {'line': 'one', 'vector': np.array([0.0, 0.0])},
        {'line': 'two', 'vector': np.array([0.1, 0.0])},
    ]
    assert len(es.cluster_and_select(candidates, k=1, per_cluster=2)) == 2


def test_cluster_and_select_claims_poems_across_clusters_not_just_within():
    # Two well-separated clusters whose nearest members both come from p1.
    candidates = [
        _cand('a', 'p1', [0.0, 0.0]),
        _cand('b', 'p2', [0.3, 0.0]),
        _cand('c', 'p1', [10.0, 10.0]),
        _cand('d', 'p3', [10.3, 10.0]),
    ]
    chosen = es.cluster_and_select(candidates, k=2, per_cluster=1)
    assert len({c['poem_id'] for c in chosen}) == 2


# ── select_exemplars: end-to-end regression on the reported bug ────────────────

SHOPOV = 'Ацо Шопов'


def test_select_exemplars_returns_no_duplicate_lines():
    # The bug as logged: Шопов's exemplar block contained 'Треба да бидеме
    # подобри.' and 'Вечер е тиха, дремлива,' twice each, teaching the model
    # that verbatim repetition was the target author's rhythm.
    for author in (SHOPOV, KONESKI, RACIN):
        lines = [e['line'] for e in es.select_exemplars(author, k=5, per_cluster=2)]
        assert len(lines) == len(set(lines)), f'duplicate exemplar line for {author}'


def test_select_exemplars_returns_one_line_per_source_poem():
    for author in (SHOPOV, KONESKI, RACIN):
        ids = [e['poem_id'] for e in es.select_exemplars(author, k=5, per_cluster=2)]
        assert len(ids) == len(set(ids)), f'two exemplars from one poem for {author}'


def test_select_exemplars_reports_poem_id_for_audit():
    # song_title alone can't tell 'Бура' from 'БУРА' in the run logs.
    exemplars = es.select_exemplars(KONESKI, k=5, per_cluster=2)
    assert all(e['poem_id'] for e in exemplars)
