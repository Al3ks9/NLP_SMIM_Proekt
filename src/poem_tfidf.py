"""
Per-poem TF-IDF (LLM style-transfer pipeline, step 2).

Distinct from tfidf_authors.py: that script's document is one author's whole
corpus and its tokens are raw regex-tokenized surface words. Here the document
is a single poem, the collection is the rest of the corpus, and the tokens are
pos_tagged.csv LEMMAS filtered to content POS — same lemma/POS conventions the
rest of the pipeline (author_vocab.json, candidate_selection.py) already uses.

Output: data/poem_tfidf_results.csv, columns author, song_title, rank, lemma,
tfidf_score — 'lemma' (not 'word') because that is honestly what this is,
distinct from tfidf_results.csv's surface-form 'word' column.

Run once (or whenever pos_tagged.csv is rebuilt):
    uv run python src/poem_tfidf.py

top_content_words() reads the CSV back for callers (llm_style_transfer.py) —
never recomputed inline per request.
"""

import csv
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

TOP_N = 10
KEEP_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


# ── Core computation (pure — no file I/O) ──────────────────────────────────────

def build_documents(pos_rows, keep_pos=KEEP_POS) -> dict:
    """
    (author, song_title) -> [lemma, lemma, ...] in corpus order, content POS only.

    CAVEAT: (author, song_title) is not a unique poem key in this corpus — ~15
    pairs (e.g. Конески's 4 poems titled 'ПЕСНА') are shared by multiple
    distinct poems, and pos_tagged.csv itself has no finer-grained id to split
    them on (inherited from pos_tag_corpus.py's schema, shared by every other
    pipeline stage — see CLAUDE.md's Data quality context). For those pairs,
    this merges the distinct poems' tokens into one TF-IDF document, so
    top_content_words() returns keywords blended across them rather than one
    poem's own. Fixing this needs a poem-level id threaded through the tagging
    pipeline, which is out of scope here.
    """
    documents = defaultdict(list)
    for r in pos_rows:
        if r['pos'] in keep_pos:
            documents[(r['author'], r['song_title'])].append(r['lemma'])
    return dict(documents)


def compute_tfidf(documents: dict) -> dict:
    """
    {(author, title): [lemma, ...]} in -> {(author, title): [(lemma, score), ...]} out,
    each document's words sorted by descending TF-IDF score (ties broken by lemma).

    Every poem is its own document; the rest of the corpus is the collection.
    Words absent from a document score 0 there and are dropped from its list —
    this is what makes the result "this poem's distinctive words," not a global
    vocabulary padded with zeros.
    """
    keys = sorted(documents.keys())
    corpus = [' '.join(documents[k]) for k in keys]

    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None)
    matrix = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()

    results = {}
    for i, key in enumerate(keys):
        row = matrix[i].toarray().flatten()
        nonzero = [(vocab[j], round(float(row[j]), 4)) for j in row.nonzero()[0]]
        nonzero.sort(key=lambda ws: (-ws[1], ws[0]))
        results[key] = nonzero
    return results


# ── File I/O ────────────────────────────────────────────────────────────────────

def load_pos_rows() -> list:
    def build():
        with open(DATA / 'pos_tagged.csv', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    return _cached('pos_rows', build)


def build_and_save(top_n: int = TOP_N) -> None:
    """Compute per-poem TF-IDF over the full corpus and write poem_tfidf_results.csv."""
    documents = build_documents(load_pos_rows())
    scores = compute_tfidf(documents)

    with open(DATA / 'poem_tfidf_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['author', 'song_title', 'rank', 'lemma', 'tfidf_score'])
        for (author, title), words in sorted(scores.items()):
            for rank, (lemma, score) in enumerate(words[:top_n], 1):
                writer.writerow([author, title, rank, lemma, score])

    print(f'Saved poem_tfidf_results.csv ({len(scores)} poems)')


def top_content_words(author: str, song_title: str, n: int = TOP_N) -> list:
    """A poem's top-N content lemmas, read from poem_tfidf_results.csv."""
    def build():
        rows = defaultdict(list)
        with open(DATA / 'poem_tfidf_results.csv', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows[(row['author'], row['song_title'])].append(
                    (int(row['rank']), row['lemma'])
                )
        return {k: [lemma for _, lemma in sorted(v)] for k, v in rows.items()}

    by_poem = _cached('poem_tfidf', build)
    return by_poem.get((author, song_title), [])[:n]


if __name__ == '__main__':
    build_and_save()
