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


def _norm_line(line: str) -> str:
    """Dedup key for a line: case-folded, whitespace-collapsed."""
    return ' '.join(line.lower().split())


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
    candidates: [{'vector': np.ndarray, 'poem_id': str, ...}, ...]. k-means over
    the vectors, then up to per_cluster candidates closest to each cluster
    centroid, tagged with their cluster id. k reduces to len(candidates) when
    there are fewer candidates than clusters requested.

    One line per source poem, enforced globally across clusters rather than
    per-cluster: the exemplar block is a tone reference, and a second line from
    an already-represented poem narrows it for no gain (Шопов's cluster 3 used
    to take both its lines from 'Notre Dame'). A candidate whose poem_id an
    earlier pick claimed is skipped, so a cluster with nothing left to offer
    returns fewer than per_cluster lines instead of repeating a poem. Across
    the corpus's 38 authors that costs 7 exemplar lines in total (218 -> 211)
    and no author drops below 5.

    Clusters are *processed* smallest-first so the constrained ones claim before
    the roomy ones — Шопов's single-member cluster 3 would otherwise lose its
    only poem to cluster 0's eight members. The returned list is still ordered
    by cluster id, so processing order isn't observable.

    Candidates without a poem_id neither claim nor are skipped, which keeps the
    function usable on bare {'line', 'vector'} dicts.
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

    by_cluster = {cluster_id: [c for c, l in zip(candidates, labels) if l == cluster_id]
                  for cluster_id in range(k)}

    chosen = []
    claimed_poems = set()
    for cluster_id in sorted(by_cluster, key=lambda c: len(by_cluster[c])):
        center = km.cluster_centers_[cluster_id]
        members = sorted(by_cluster[cluster_id],
                        key=lambda c: float(np.linalg.norm(c['vector'] - center)))

        picked = 0
        for cand in members:
            if picked >= per_cluster:
                break
            poem_id = cand.get('poem_id')
            if poem_id is not None:
                if poem_id in claimed_poems:
                    continue
                claimed_poems.add(poem_id)
            chosen.append({**cand, 'cluster': cluster_id})
            picked += 1

    chosen.sort(key=lambda c: c['cluster'])
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
    Up to max_lines_per_poem *distinct* lines per poem by target_author that
    contain at least one word from wanted_words. Returns
    [{'line', 'poem_id', 'song_title', 'poem_tokens'}, ...] — poem_tokens
    carried along so callers can align the line to its lemma stream without a
    second lookup.

    Line texts are de-duplicated author-wide (case- and whitespace-insensitive),
    which closes both routes duplicate exemplars took into the prompt:

    - a refrain repeated inside one poem used to spend the whole per-poem quota
      on one text (Шопов's 'Треба да бидеме подобри' opens with the same line
      twice; 'Партизанска колона' repeats 'Вечер е тиха, дремлива,' at lines 2
      and 21). A repeat is skipped without consuming a slot, so the poem still
      contributes max_lines_per_poem distinct lines.
    - the corpus's near-duplicate poems each contributed the same line
      independently (Конески's 'Бура' poem_id=13 and 'БУРА' poem_id=566 are the
      same poem under two title casings) — hence author-wide, not per-poem.

    O(poems_by_author x corpus): corpus_token_stream() scans all of pos_rows per
    poem. Fine at this corpus's scale (~2s cold for a full select_exemplars()
    call); an index keyed by poem_id would be the fix if this corpus grows
    enough for it to matter.
    """
    pos_rows = load_pos_rows()
    out = []
    seen_lines = set()
    for row in load_stripped_songs():
        if row['author'] != target_author:
            continue
        title = row['song_title']
        poem_tokens = corpus_token_stream(row['poem_id'], pos_rows)
        taken = 0
        for line in row['song_text'].splitlines():
            line = line.strip()
            if not line or taken >= max_lines_per_poem:
                continue
            if not wanted_words & set(_words(line)):
                continue
            norm = _norm_line(line)
            if norm in seen_lines:
                continue
            seen_lines.add(norm)
            out.append({'line': line, 'poem_id': row['poem_id'],
                       'song_title': title, 'poem_tokens': poem_tokens})
            taken += 1
    return out


def candidate_pool(target_author: str, max_lines_per_poem: int = DEFAULT_MAX_LINES_PER_POEM) -> list:
    """
    Qualifying lines, embedded and ready for clustering.
    [{'line', 'author', 'poem_id', 'song_title', 'vector'}, ...] — lines whose
    content words have no embeddable lemma at all are dropped.
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
                        'poem_id': cand['poem_id'],
                        'song_title': cand['song_title'], 'vector': vec})
    return pool


def select_exemplars(target_author: str, k: int = DEFAULT_K,
                     per_cluster: int = DEFAULT_PER_CLUSTER,
                     max_lines_per_poem: int = DEFAULT_MAX_LINES_PER_POEM) -> list:
    """
    Full step-4 path: target author in, ~k*per_cluster distinct exemplar lines
    out — no repeated line text, and one line per source poem — each tagged with
    its source poem for audit (strip that attribution before showing lines to
    the model — see llm_style_transfer.assemble_prompt).

    poem_id is reported alongside song_title because the title alone can't tell
    the corpus's same-titled poems apart in a run log ('Бура' vs 'БУРА').
    """
    pool = candidate_pool(target_author, max_lines_per_poem)
    chosen = cluster_and_select(pool, k=k, per_cluster=per_cluster)
    return [{'line': c['line'], 'author': c['author'], 'poem_id': c['poem_id'],
            'song_title': c['song_title'], 'cluster': c['cluster']}
           for c in chosen]
