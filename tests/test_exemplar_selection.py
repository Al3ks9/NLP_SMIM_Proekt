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
    counts = Counter(e['song_title'] for e in exemplars)
    assert all(c <= 2 for c in counts.values())


def test_select_exemplars_covers_multiple_poems():
    # per_cluster capped lines from a single poem would defeat the "force
    # multi-poem coverage" point of the 2-line cap.
    exemplars = es.select_exemplars(KONESKI, k=5, per_cluster=2)
    titles = {e['song_title'] for e in exemplars}
    assert len(titles) > 1
