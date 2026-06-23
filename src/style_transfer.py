"""
Style transfer: rewrite a poem in the style of a target Macedonian poet.

Usage:
    uv run python style_transfer.py                    # interactive demo
    uv run python style_transfer.py "text" "Author"    # CLI mode
"""

import csv
import json
import pickle
import sys
import random
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import spacy
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}  # PROPN excluded — never replace proper nouns

# ── Load resources ────────────────────────────────────────────────────────────

print("Loading resources...", flush=True)

nlp = spacy.load('mk_core_news_lg')
G = nx.read_gexf(MODELS / 'skg_final.gexf')

with open(MODELS / 'author_vocab.json', encoding='utf-8') as f:
    author_vocab = json.load(f)

# Build surface form lookup: (author, lemma, pos) → most common surface word
# This lets us insert grammatically natural word forms, not bare lemmas
surface_counter: dict = defaultdict(Counter)
pos_rows = []
with open(DATA / 'pos_tagged.csv', encoding='utf-8') as f:
    pos_rows = list(csv.DictReader(f))
for r in pos_rows:
    surface_counter[(r['author'], r['lemma'], r['pos'])][r['word']] += 1


def get_surface(author: str, lemma: str, pos: str) -> str:
    """Most common surface form used by author for (lemma, pos). Falls back to lemma."""
    key = (author, lemma, pos)
    if key in surface_counter:
        return surface_counter[key].most_common(1)[0][0]
    return lemma


def get_surface_morph(author: str, lemma: str, pos: str, source_token) -> str:
    """
    Return the surface form of (lemma, pos) for author whose ending matches
    the source token's suffix. In Macedonian, word endings encode grammatical
    form (person, number, tense, definiteness, gender), so suffix matching
    gives a good morphological proxy without a dedicated morphological analyzer.

    Tries 3-char suffix first, then 2-char, then falls back to most common form.
    """
    src = source_token.text.lower()
    lemma_morph = morph_lookup.get(author, {}).get(lemma, {}).get(pos, {})

    for n in (3, 2):
        suffix = src[-n:] if len(src) >= n else src
        surface = lemma_morph.get(suffix)
        if surface:
            return surface

    return get_surface(author, lemma, pos)


# Map TF-IDF surface words → lemmas
word_lemma_counter: dict = defaultdict(Counter)
for r in pos_rows:
    word_lemma_counter[r['word']][r['lemma']] += 1
word_to_lemma = {w: c.most_common(1)[0][0] for w, c in word_lemma_counter.items()}

tfidf_lemmas: dict = defaultdict(set)  # author → set of distinctive lemmas
with open(DATA / 'tfidf_results.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        lemma = word_to_lemma.get(row['word'], row['word'])
        tfidf_lemmas[row['author']].add(lemma)

with open(MODELS / 'classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open(MODELS / 'word_embeddings.json', encoding='utf-8') as f:
    word_embeddings: dict = {k: np.array(v) for k, v in json.load(f).items()}

with open(MODELS / 'morph_lookup.json', encoding='utf-8') as f:
    morph_lookup: dict = json.load(f)

print("Ready.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def predict_author(text: str) -> tuple[str, float]:
    proba = classifier.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    return classifier.classes_[idx], round(float(proba[idx]), 4)


def _poem_context_nodes(text: str) -> set[str]:
    """
    Return all lemma_POS SKG nodes found in the poem.
    Used to score candidates by how well they fit the poem's semantic space.
    """
    doc = nlp(text)
    return {
        f'{token.lemma_}_{token.pos_}'
        for token in doc
        if token.pos_ in CONTENT_POS and f'{token.lemma_}_{token.pos_}' in G
    }


def _score(candidate_lemma: str, pos: str, target_author: str,
           orig_lemma: str, freq: float, max_freq: float,
           context_nodes: set, used_count: Counter) -> float:
    score = freq / max_freq                                      # normalized frequency [0, 1]

    if candidate_lemma in tfidf_lemmas.get(target_author, set()):
        score += 3.0                                             # target-distinctive bonus

    # Semantic similarity to the original word via embeddings
    if orig_lemma in word_embeddings and candidate_lemma in word_embeddings:
        sim = float(np.dot(word_embeddings[orig_lemma], word_embeddings[candidate_lemma]))
        score += sim * 2.0                                       # semantic similarity weight

    # Context compatibility via SKG co-occurrence
    node_id = f'{candidate_lemma}_{pos}'
    if node_id in G:
        for ctx in context_nodes:
            if G.has_edge(node_id, ctx):
                score += G[node_id][ctx].get('weight', 1) * 0.1

    score -= used_count[candidate_lemma] * 2.0                   # diversity penalty
    return score


# ── Core transfer function ────────────────────────────────────────────────────

def transfer_style(text: str, target_author: str, source_author: str = None) -> str:
    """
    Rewrite text in target_author's style.

    Eligibility to replace a token:
      (a) Its lemma is in source TF-IDF top-20 AND not in target TF-IDF top-20
          → swapping a true source style-marker for a target one
      (b) Its lemma is completely absent from target's vocabulary for that POS
          → the target simply never uses this word

    Replacements use the target author's most common surface form of the chosen
    lemma, so the output is morphologically natural rather than bare lemmas.
    """
    if target_author not in author_vocab:
        raise ValueError(
            f"Author not found. Available:\n{', '.join(sorted(author_vocab.keys()))}"
        )

    src_tfidf = tfidf_lemmas.get(source_author, set()) if source_author else set()
    tgt_tfidf = tfidf_lemmas.get(target_author, set())

    # Per-POS vocabulary sets for the target author (fast membership check)
    tgt_vocab: dict = {
        pos: {lemma for lemma, _ in words}
        for pos, words in author_vocab.get(target_author, {}).items()
    }

    context_nodes = _poem_context_nodes(text)
    used_count: Counter = Counter()
    output_lines = []

    for line in text.splitlines():
        if not line.strip():
            output_lines.append(line)
            continue

        doc = nlp(line)
        rebuilt = []

        for token in doc:
            if token.pos_ not in CONTENT_POS or token.is_punct or token.is_space:
                rebuilt.append(token.text_with_ws)
                continue

            # Skip place-derived adjectives (e.g. битолско, македонски, градска)
            if token.pos_ == 'ADJ' and token.text.lower().endswith(('ски', 'ска', 'ско', 'ски', 'цки', 'цка', 'цко')):
                rebuilt.append(token.text_with_ws)
                continue

            pos = token.pos_
            lemma = token.lemma_
            in_tgt_vocab = lemma in tgt_vocab.get(pos, set())

            # Eligibility: only replace genuine style markers or words foreign to target
            should_replace = (
                (lemma in src_tfidf and lemma not in tgt_tfidf)  # (a)
                or (not in_tgt_vocab)                             # (b)
            )
            if not should_replace:
                rebuilt.append(token.text_with_ws)
                continue

            candidates = author_vocab.get(target_author, {}).get(pos, [])
            if not candidates:
                rebuilt.append(token.text_with_ws)
                continue

            max_freq = candidates[0][1]
            scored = [
                (_score(cand, pos, target_author, lemma, freq, max_freq,
                        context_nodes, used_count), cand)
                for cand, freq in candidates
                if cand != lemma
            ]
            if not scored:
                rebuilt.append(token.text_with_ws)
                continue

            scored.sort(key=lambda x: -x[0])
            top3 = scored[:3]

            # Softmax over top-3 for natural variety
            vals = np.array([s for s, _ in top3], dtype=float)
            vals -= vals.max()
            probs = np.exp(vals) / np.exp(vals).sum()
            best_lemma = top3[np.random.choice(len(top3), p=probs)][1]

            surface = get_surface_morph(target_author, best_lemma, pos, token)
            used_count[best_lemma] += 1
            rebuilt.append(surface + token.whitespace_)

        output_lines.append(''.join(rebuilt))

    return '\n'.join(output_lines)


# ── CLI / interactive ─────────────────────────────────────────────────────────

def run(text: str, target_author: str, source_author: str = None) -> None:
    orig_author, orig_conf = predict_author(text)
    print(f"Original — classifier: '{orig_author}' ({orig_conf:.1%})\n")
    print("── Original ──────────────────────────────────")
    print(text)

    transferred = transfer_style(text, target_author, source_author=source_author)

    new_author, new_conf = predict_author(transferred)
    print(f"\n── Transferred → {target_author} ──────────────")
    print(transferred)
    print(f"\nClassifier after transfer: '{new_author}' ({new_conf:.1%})")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    else:
        songs = []
        with open(DATA / 'stripped_songs.csv', encoding='utf-8') as f:
            songs = list(csv.DictReader(f))

        source_author = 'Кочо Рацин'
        target_author = 'Блаже Конески'

        sample = next((r for r in songs if r['author'] == source_author), songs[0])
        text = sample['song_text'].strip()
        print(f"Source: '{sample['song_title']}' by {sample['author']}\n")
        run(text, target_author, source_author=source_author)
