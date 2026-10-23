"""
Style transfer: rewrite a poem in the style of a target Macedonian poet.

Usage:
    uv run python style_transfer.py                    # interactive demo
    uv run python style_transfer.py "text" "Author"    # CLI mode
"""

import csv
import pickle
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

import candidate_selection
import tagging

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}  # PROPN excluded — never replace proper nouns

# ── Load resources ────────────────────────────────────────────────────────────
#
# Deferred into a function rather than run at module scope: candidate_selection's
# resources include word_embeddings.json (299 MB, gitignored), and classifier.pkl
# is also a pipeline output that need not exist for every caller. Pure functions
# like splice() — and tests that only exercise them — must be importable without
# any of this on disk. The CLI entry point below calls load_resources() before
# doing anything that needs it.

author_vocab = pos_rows = classifier = None
resources: dict = {}
tfidf_lemmas: dict = {}


def load_resources() -> None:
    """Load the classifier and candidate_selection's resource bundle into module
    globals. Must run before predict_author/transfer_style are called.

    author_vocab and pos_rows come from candidate_selection's cache rather than
    being read again here, so both modules are guaranteed to rank against the
    same vocabulary and the same tagging.
    """
    global author_vocab, pos_rows, classifier, resources, tfidf_lemmas

    print("Loading resources...", flush=True)

    resources = candidate_selection.load_resources()
    author_vocab = resources['author_vocab']
    pos_rows = candidate_selection.load_pos_rows()

    # Map TF-IDF surface words → lemmas. tfidf_results.csv is keyed on surface
    # words, but should_replace() below compares lemmas.
    word_lemma_counter: dict = defaultdict(Counter)
    for r in pos_rows:
        word_lemma_counter[r['word']][r['lemma']] += 1
    word_to_lemma = {w: c.most_common(1)[0][0] for w, c in word_lemma_counter.items()}

    tfidf_lemmas = defaultdict(set)  # author → set of distinctive lemmas
    with open(DATA / 'tfidf_results.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lemma = word_to_lemma.get(row['word'], row['word'])
            tfidf_lemmas[row['author']].add(lemma)

    with open(MODELS / 'classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    print("Ready.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def predict_author(text: str) -> tuple[str, float]:
    proba = classifier.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    return classifier.classes_[idx], round(float(proba[idx]), 4)


def splice(line: str, edits: list) -> str:
    """Write surfaces into the original line at recorded character offsets.

    Rebuilding from the source text rather than from tagger tokens makes
    out-of-region edits structurally impossible and preserves spacing and
    punctuation exactly.
    """
    out, prev = [], 0
    for start, end, surface in sorted(edits):
        out.append(line[prev:start])
        out.append(surface)
        prev = end
    out.append(line[prev:])
    return ''.join(out)


# Criterion (b) — "the target author never uses this word" — applies only here.
# NOUN is excluded on purpose, for two independent reasons:
#
#   Content. The criterion asks whether a lemma is missing from the target's
#   vocabulary for its POS. On an adjective that is a statement about how the
#   author qualifies things; on a noun it is a statement about what he wrote
#   about. Конески's NOUN vocabulary is 1,297 lemmas, so leaving NOUN in meant
#   replacing every noun outside those 1,297 — which is how врба and војник,
#   the subject of the Рацин stanza, became вик and сокол.
#
#   Agreement. Gender is inflection on ADJ/ADV/VERB and lexical on NOUN (26% of
#   ADJ lemmas are attested in more than one gender against 3% of NOUN lemmas),
#   so morph.surface_form can realise a replacement adjective into the slot's
#   own gender but can never do that for a noun. 25.2% of noun tokens sit beside
#   an agreeing adjective, and a replacement drawn from Конески's 43/38/19
#   Fem/Masc/Neut noun vocabulary keeps the source gender only 36.4% of the
#   time — so ~64% of those replacements break agreement with an adjective this
#   module never touched and therefore cannot repair.
#
# Per-token the loss is small: author-identification accuracy per 100 tokens is
# 2.79 for ADJ and 4.58 for ADV against 1.85 for NOUN (5-fold CV, 24 authors,
# 1,169 poems, 0.145 baseline). Nouns lead on raw accuracy only through volume.
#
# Nouns remain reachable through criterion (a), which is what carries the real
# noun transfers — the dialect variants огин/оган and раца/рака.
B_ELIGIBLE_POS = {'ADJ', 'ADV', 'VERB'}


def should_replace(lemma: str, pos: str, src_tfidf: set,
                   tgt_tfidf: set, tgt_vocab: dict) -> bool:
    """Is this token a replacement target?

      (a) Its lemma is distinctive to the source and not to the target
          → swapping a measured source style-marker for a target one.
      (b) Its lemma is absent from the target's vocabulary for that POS, and
          that POS is one where absence means style rather than subject matter
          → see B_ELIGIBLE_POS.
    """
    if lemma in src_tfidf and lemma not in tgt_tfidf:                 # (a)
        return True
    return pos in B_ELIGIBLE_POS and lemma not in tgt_vocab.get(pos, set())  # (b)


def _pick(candidates: list, used_count: Counter) -> dict:
    """Choose one of candidate_selection's ranked candidates for this slot.

    Deterministic, unlike the softmax-over-top-3 sampling this replaced. Those
    probabilities were computed over scores spread across a 0–5 range; on
    candidate_selection's 0–1 composite the same softmax is close to uniform, so
    sampling would have amounted to picking at random among the top three.

    The one thing sampling did buy — not repeating a lemma across a poem — is
    kept explicitly: an unused lemma outranks a used one regardless of score.
    """
    return min(candidates,
               key=lambda c: (used_count[c['lemma']], -c['score'], c['lemma']))


# ── Core transfer function ────────────────────────────────────────────────────

def transfer_style(text: str, target_author: str, source_author: str = None) -> str:
    """
    Rewrite text in target_author's style.

    Ranking and inflection are candidate_selection.build_slot()'s job, not this
    module's: the same per-slot scoring and the same relaxation tiers that build
    the LLM probe's candidate lists decide the replacement here. This module owns
    only *which* tokens are eligible (should_replace) and *which* of the ranked
    candidates to take (_pick).
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

    used_count: Counter = Counter()
    output_lines = []

    verb_lemmas = tagging.load_verb_lemmas(pos_rows)
    tokens = tagging.tag_lines(text, verb_lemmas)
    by_line: dict = defaultdict(list)
    for t in tokens:
        by_line[t.line].append(t)

    for li, line in enumerate(text.splitlines()):
        if not line.strip():
            output_lines.append(line)
            continue

        edits = []

        # by_line holds only content tokens, so the neighbours of a slot are its
        # neighbouring *content* tokens — which is what pos_transitions.json was
        # counted over, and what candidate_selection.slot_contexts() feeds it.
        line_tokens = by_line[li]

        for i, token in enumerate(line_tokens):
            # tag_lines() also yields PROPN (its content-POS set is a superset
            # of CONTENT_POS) — proper nouns are never replacement targets.
            if token.pos not in CONTENT_POS:
                continue

            # Skip place-derived adjectives (e.g. битолско, македонски, градска)
            if token.pos == 'ADJ' and token.text.endswith(('ски', 'ска', 'ско', 'ски', 'цки', 'цка', 'цко')):
                continue

            pos = token.pos
            lemma = token.lemma

            if not should_replace(lemma, pos, src_tfidf, tgt_tfidf, tgt_vocab):
                continue

            slot = candidate_selection.build_slot(
                token.text, lemma, pos, token.feats,
                line_tokens[i - 1].pos if i > 0 else None,
                line_tokens[i + 1].pos if i + 1 < len(line_tokens) else None,
                target_author, resources,
            )
            if not slot['candidates']:
                continue

            chosen = _pick(slot['candidates'], used_count)
            used_count[chosen['lemma']] += 1
            edits.append((token.start_char, token.end_char, chosen['surface']))

        output_lines.append(splice(line, edits))

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
    load_resources()
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
