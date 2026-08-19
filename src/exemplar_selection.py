"""
Exemplar line selector (LLM style-transfer pipeline, step 4).

For a target author: pull lines containing at least one of their top-20 TF-IDF
words (tfidf_results.csv), cap 2 lines per source poem to force multi-poem
coverage, embed each line as the centroid of its content-word vectors
(word_embeddings.json), k-means cluster the centroids (k~=5, configurable), and
take the per_cluster lines closest to each cluster centroid (~8-10 lines total
at the defaults) as tone/rhythm exemplars for the LLM prompt.

Reuses candidate_selection's corpus-alignment machinery (corpus_token_stream,
_align) rather than re-deriving line->lemma alignment — that greedy alignment
logic already exists and is exercised there; duplicating it here would risk
drift between the two.
"""

import csv
import re
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from candidate_selection import (  # noqa: F401 — reused, see module docstring
    _align, corpus_token_stream, load_pos_rows, load_tfidf, load_word_embeddings,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

TOP_TFIDF_WORDS = 20
DEFAULT_K = 5
DEFAULT_PER_CLUSTER = 2
DEFAULT_MAX_LINES_PER_POEM = 2
CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def _words(text: str) -> list:
    return re.findall(r'\w+', text.lower())


# ── Line embedding ───────────────────────────────────────────────────────────────

def line_embedding(lemmas: list, embeddings: dict):
    """Centroid of the embeddable lemmas' vectors, or None if none are embeddable."""
    vectors = [embeddings[l] for l in lemmas if l in embeddings]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


# ── Clustering + representative selection ───────────────────────────────────────

def cluster_and_select(candidates: list, k: int = DEFAULT_K,
                       per_cluster: int = DEFAULT_PER_CLUSTER) -> list:
    """
    candidates: [{'vector': np.ndarray, ...}, ...]. k-means over the vectors,
    then per_cluster candidates closest to each cluster centroid, tagged with
    their cluster id. k reduces to len(candidates) when there are fewer
    candidates than clusters requested.
    """
    if not candidates:
        return []

    n = len(candidates)
    k = min(k, n)
    # float64: word_embeddings.json vectors are float32, and sklearn's squared-
    # distance trick (x^2 - 2xy + y^2) is known to overflow/underflow at that
    # precision for some inputs — float64 keeps KMeans numerically stable here.
    vectors = np.stack([c['vector'] for c in candidates]).astype(np.float64)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    with warnings.catch_warnings():
        # Apple Accelerate BLAS (aarch64 macOS) raises spurious divide-by-zero /
        # overflow / invalid-value RuntimeWarnings out of plain matmul during
        # k-means++ init — verified the centers/labels stay finite and correct
        # (see commit message); this is a platform BLAS quirk, not a real
        # numerical fault, so it's suppressed rather than chased further.
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
        labels = km.fit_predict(vectors)

    chosen = []
    for cluster_id in range(k):
        members = [(i, candidates[i]) for i in range(n) if labels[i] == cluster_id]
        center = km.cluster_centers_[cluster_id]
        members.sort(key=lambda iv: float(np.linalg.norm(iv[1]['vector'] - center)))
        for i, cand in members[:per_cluster]:
            chosen.append({**cand, 'cluster': cluster_id})
    return chosen


# ── Candidate line collection ─────────────────────────────────────────────────

def load_stripped_songs() -> list:
    def build():
        with open(DATA / 'stripped_songs.csv', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    return _cached('stripped_songs', build)


def top_tfidf_words(target_author: str, n: int = TOP_TFIDF_WORDS) -> set:
    """Target author's top-N TF-IDF surface words, as a set for membership tests."""
    author_scores = load_tfidf().get(target_author, {})
    ranked = sorted(author_scores.items(), key=lambda ws: -ws[1])
    return {w for w, _ in ranked[:n]}


def qualifying_lines(target_author: str, wanted_words: set,
                     max_lines_per_poem: int = DEFAULT_MAX_LINES_PER_POEM) -> list:
    """
    Up to max_lines_per_poem lines per poem by target_author that contain at
    least one word from wanted_words. Returns
    [{'line', 'song_title', 'poem_tokens'}, ...] — poem_tokens carried along so
    callers can align the line to its lemma stream without a second lookup.

    O(poems_by_author x corpus): corpus_token_stream() scans all of pos_rows per
    poem. Fine at this corpus's scale (~2s cold for a full select_exemplars()
    call); an index keyed by (author, song_title) would be the fix if this
    corpus grows enough for it to matter.
    """
    pos_rows = load_pos_rows()
    out = []
    for row in load_stripped_songs():
        if row['author'] != target_author:
            continue
        title = row['song_title']
        poem_tokens = corpus_token_stream(row['poem_id'], pos_rows)
        taken = 0
        for line in row['song_text'].splitlines():
            if not line.strip() or taken >= max_lines_per_poem:
                continue
            if wanted_words & set(_words(line)):
                out.append({'line': line.strip(), 'song_title': title,
                           'poem_tokens': poem_tokens})
                taken += 1
    return out


def candidate_pool(target_author: str, max_lines_per_poem: int = DEFAULT_MAX_LINES_PER_POEM) -> list:
    """
    Qualifying lines, embedded and ready for clustering.
    [{'line', 'author', 'song_title', 'vector'}, ...] — lines whose content
    words have no embeddable lemma at all are dropped.
    """
    embeddings = load_word_embeddings()
    wanted = top_tfidf_words(target_author)

    pool = []
    for cand in qualifying_lines(target_author, wanted, max_lines_per_poem):
        mapping = _align(cand['line'], cand['poem_tokens'])
        lemmas = [cand['poem_tokens'][ti]['lemma'] for ti in mapping.values()
                 if cand['poem_tokens'][ti]['pos'] in CONTENT_POS]
        vec = line_embedding(lemmas, embeddings)
        if vec is not None:
            pool.append({'line': cand['line'], 'author': target_author,
                        'song_title': cand['song_title'], 'vector': vec})
    return pool


def select_exemplars(target_author: str, k: int = DEFAULT_K,
                     per_cluster: int = DEFAULT_PER_CLUSTER,
                     max_lines_per_poem: int = DEFAULT_MAX_LINES_PER_POEM) -> list:
    """
    Full step-4 path: target author in, ~k*per_cluster diverse exemplar lines
    out, each tagged with its source poem for audit (strip that attribution
    before showing lines to the model — see llm_style_transfer.assemble_prompt).
    """
    pool = candidate_pool(target_author, max_lines_per_poem)
    chosen = cluster_and_select(pool, k=k, per_cluster=per_cluster)
    return [{'line': c['line'], 'author': c['author'],
            'song_title': c['song_title'], 'cluster': c['cluster']}
           for c in chosen]
