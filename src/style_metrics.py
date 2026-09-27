"""
Per-generation quality metrics for style-transfer output (Qwen SFT pipeline,
stage 7): degeneracy, language/format compliance, structural fit, target-style
match, and source-content preservation.

Deliberately one module of small pure functions over (source_text,
generated_text, target_author), aggregated nowhere: compare_generations.py
averages them across a run for the base-vs-tuned A/B, and the GRPO stage this
pipeline is headed for needs the *same* numbers per candidate as a reward
signal. A reward that disagreed with the evaluation metric would be a bug, so
there is one implementation of each.

Every metric is oriented so that higher is better and, where it is bounded,
lands in [0, 1] -- so a reward can be a plain weighted sum without remembering
which components need flipping. The two-sided structural deltas
(llm_style_transfer.validate_structure) are the exception; structure_fit()
converts them into that orientation.

No GPU and no LLM call: char n-gram stylometry (models/classifier.pkl),
author TF-IDF vocabulary (data/tfidf_results.csv) and plain string statistics
only, so scoring a full validation run is seconds of CPU.
"""

import csv
import pickle
import re
from collections import Counter
from pathlib import Path

import llm_style_transfer as lst

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'

CLASSIFIER_PATH = MODELS / 'classifier.pkl'
TFIDF_PATH = DATA / 'tfidf_results.csv'

# Content-word proxy for the overlap metrics. The generated text is not POS
# tagged (tagging.py runs over the corpus, not over model output), so length
# stands in for "content word" -- Macedonian function words are overwhelmingly
# short (и, во, на, се, од, со, за, да, не, ја, го, ми, ти).
MIN_CONTENT_LEN = 4

# Structural deltas are absolute counts; a miss is scored relative to the
# target's own magnitude, floored so tiny targets (~2 stanzas) don't make any
# miss look catastrophic.
_STRUCT_FLOOR = {'lines': 4.0, 'stanzas': 1.0, 'tokens_per_line': 1.0,
                 'lines_per_stanza': 1.0}

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


# ── Tokenization helpers ────────────────────────────────────────────────────────

_WORD_RE = re.compile(r'[^\W\d_]+', re.UNICODE)


def tokens(text: str) -> list:
    """Lowercased word tokens, no digits or punctuation."""
    return _WORD_RE.findall(text.lower())


def lines(text: str) -> list:
    return [l.strip() for l in text.strip().splitlines() if l.strip()]


def content_types(text: str) -> set:
    return {t for t in tokens(text) if len(t) >= MIN_CONTENT_LEN}


# ── 1. Degeneracy / repetition ──────────────────────────────────────────────────

def _distinct_n(toks: list, n: int) -> float:
    """Unique n-grams / total n-grams. 1.0 = nothing repeats, low = the model
    is looping. Undefined (returns 0.0) for text shorter than n tokens, which
    is itself a failure."""
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_metrics(text: str) -> dict:
    """
    The failure mode this pipeline's SFT output actually shows: whole lines
    repeated verbatim, and the same few words recombined with commas.

    distinct_1/2/3 catch the word-level looping; line_uniqueness catches the
    copied refrain. All three are "higher is better" fractions in [0, 1].
    """
    toks = tokens(text)
    poem_lines = lines(text)
    line_counts = Counter(poem_lines)
    return {
        'distinct_1': round(_distinct_n(toks, 1), 4),
        'distinct_2': round(_distinct_n(toks, 2), 4),
        'distinct_3': round(_distinct_n(toks, 3), 4),
        'line_uniqueness': round(len(line_counts) / len(poem_lines), 4) if poem_lines else 0.0,
        'max_line_repeats': max(line_counts.values()) if line_counts else 0,
        'num_tokens': len(toks),
    }


# ── 2. Language / format compliance ─────────────────────────────────────────────

_CYRILLIC_RE = re.compile(r'[Ѐ-ӿ]')
_LATIN_RE = re.compile(r'[A-Za-z]')
# Chat/markdown scaffolding a model emits when it answers *about* the task
# instead of just writing the poem -- the base model's characteristic failure.
_MARKUP_RE = re.compile(r'(\*\*|^#{1,6}\s|^\s*[-*]\s|^\s*\d+\.\s|```)', re.MULTILINE)


def language_metrics(text: str) -> dict:
    """Is this a Macedonian poem at all, or an English/markdown answer about
    one? cyrillic_ratio is over letters only, so punctuation and line breaks
    don't dilute it."""
    cyr = len(_CYRILLIC_RE.findall(text))
    lat = len(_LATIN_RE.findall(text))
    letters = cyr + lat
    return {
        'cyrillic_ratio': round(cyr / letters, 4) if letters else 0.0,
        'is_cyrillic': 1.0 if letters and cyr / letters >= 0.95 else 0.0,
        'no_markup': 0.0 if _MARKUP_RE.search(text) else 1.0,
        'is_nonempty': 1.0 if lines(text) else 0.0,
    }


# ── 3. Structural fit to the target author ──────────────────────────────────────

def structure_fit(text: str, target_author: str) -> dict:
    """
    llm_style_transfer.validate_structure's signed deltas, plus a
    higher-is-better fit score per dimension: 1 / (1 + |delta| / scale), where
    scale is max(target value, a per-dimension floor). 1.0 = exactly on
    target, 0.5 = off by one scale unit.
    """
    targets = lst.structural_targets(target_author)
    fit = lst.validate_structure(text, targets)
    scores = {}
    for key, target_col in (('lines', 'avg_lines_per_poem'),
                            ('stanzas', 'avg_stanzas_per_poem'),
                            ('tokens_per_line', 'avg_tokens_per_line'),
                            ('lines_per_stanza', 'avg_lines_per_stanza')):
        scale = max(targets[target_col], _STRUCT_FLOOR[key])
        scores[f'fit_{key}'] = round(1.0 / (1.0 + abs(fit[f'delta_{key}']) / scale), 4)
    scores['fit_structure'] = round(sum(scores.values()) / len(scores), 4)
    return {**fit, **scores}


# ── 4. Target-style match ───────────────────────────────────────────────────────

def load_classifier():
    """models/classifier.pkl -- the char-(2,4)-gram + LinearSVC author
    classifier from classifier.py, calibrated so predict_proba works. Used
    here as a stylometric judge of the *generated* text: it was fitted on real
    poems only, so it has never seen model output and is not being asked to
    grade its own training data."""
    return _cached('classifier', lambda: pickle.loads(CLASSIFIER_PATH.read_bytes()))


def load_author_tfidf() -> dict:
    """{author: {word, ...}} from tfidf_results.csv -- each author's top
    surface words by TF-IDF, the same table llm_style_transfer's vocabulary
    palette draws on."""
    def build():
        by_author = {}
        with open(TFIDF_PATH, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                by_author.setdefault(row['author'], set()).add(row['word'].lower())
        return by_author
    return _cached('author_tfidf', build)


def style_match(text: str, target_author: str) -> dict:
    """
    Two independent readings of "does this sound like the target author":

    - clf_target_prob / clf_top1: the stylometric classifier's own verdict.
      Character n-grams respond to morphology and orthographic habit, not to
      topic, which is exactly the signal style transfer is supposed to move.
    - tfidf_hit_rate: share of the target's distinctive TF-IDF words that
      actually appear. Lexical, and the thing the Gemma prompt fed the model
      explicitly as a palette -- so it measures whether the distilled model
      kept that behaviour without being handed the list.
    """
    clf = load_classifier()
    classes = list(clf.classes_)
    result = {'clf_target_prob': 0.0, 'clf_top1': 0.0, 'clf_predicted': ''}
    if lines(text) and target_author in classes:
        probs = clf.predict_proba([text])[0]
        idx = classes.index(target_author)
        result = {
            'clf_target_prob': round(float(probs[idx]), 4),
            'clf_top1': 1.0 if classes[int(probs.argmax())] == target_author else 0.0,
            'clf_predicted': classes[int(probs.argmax())],
        }

    palette = load_author_tfidf().get(target_author, set())
    gen_tokens = set(tokens(text))
    result['tfidf_hit_rate'] = (round(len(palette & gen_tokens) / len(palette), 4)
                                if palette else 0.0)
    return result


# ── 5. Source-content preservation ──────────────────────────────────────────────

def content_preservation(source_text: str, text: str) -> dict:
    """
    Style transfer has to carry the source poem's content across, or the model
    is just writing an unrelated poem in the target's voice -- which a style
    metric alone would happily reward.

    content_recall is the share of the source's content-word types that
    survive; content_jaccard is symmetric, so a generation that dumps the whole
    source vocabulary plus padding can't max it out. Surface forms, not lemmas:
    the generated text is untagged (see MIN_CONTENT_LEN), so this undercounts
    inflectional matches equally for every run being compared.

    line_novelty is the guard on both of them. The untuned base model's
    characteristic output is the source poem reproduced verbatim under a
    markdown heading -- which scores a perfect content_recall and, because it
    is real human poetry, excellent diversity and fluency too. Without a
    copy check, "did nothing" is the highest-scoring strategy here.
    """
    src, gen = content_types(source_text), content_types(text)
    result = {'content_recall': 0.0, 'content_jaccard': 0.0,
              'line_novelty': 1.0 if lines(text) else 0.0}
    if src:
        shared, union = src & gen, src | gen
        result['content_recall'] = round(len(shared) / len(src), 4)
        result['content_jaccard'] = round(len(shared) / len(union), 4) if union else 0.0

    source_lines = {_normalize_line(l) for l in lines(source_text)}
    gen_lines = [_normalize_line(l) for l in lines(text)]
    gen_lines = [l for l in gen_lines if l]
    if gen_lines and source_lines:
        copied = sum(1 for l in gen_lines if l in source_lines)
        result['line_novelty'] = round(1.0 - copied / len(gen_lines), 4)
    return result


def _normalize_line(line: str) -> str:
    """A line reduced to its word tokens, so a copy is still detected when the
    model re-punctuates it or changes case."""
    return ' '.join(tokens(line))


# A generated line is a copy when at least this share of its word types
# occurs in a single source line. Containment, not Jaccard: with this
# corpus's ~5-token lines one swapped word gives Jaccard 4/6 = 0.67, which a
# Jaccard threshold of 0.7 would wave through as "novel".
COPY_OVERLAP_THRESHOLD = 0.75
_COPY_MIN_TOKENS = 3


def copy_novelty(source_text: str, text: str,
                 threshold: float = COPY_OVERLAP_THRESHOLD) -> float:
    """
    Share of generated lines that are NOT (near-)copies of a source line --
    the GRPO content reward's copy gate. line_novelty (exact match after
    normalisation) is kept unchanged for the reported SFT numbers; this is
    its stricter sibling, catching the one-or-two-words-changed copy that an
    embedding content reward would otherwise pay out on.

    Lines under _COPY_MIN_TOKENS tokens count as copies only on an exact
    match: containment on a two-word line is too coarse to mean anything.
    Empty generations score 0.0 (nothing novel was written).
    """
    gen_lines = [tokens(l) for l in lines(text)]
    gen_lines = [g for g in gen_lines if g]
    if not gen_lines:
        return 0.0
    src_sets = [set(tokens(l)) for l in lines(source_text)]
    src_exact = {' '.join(tokens(l)) for l in lines(source_text)}
    copied = 0
    for g in gen_lines:
        if len(g) < _COPY_MIN_TOKENS:
            copied += ' '.join(g) in src_exact
            continue
        gset = set(g)
        if any(len(gset & s) / len(gset) >= threshold for s in src_sets):
            copied += 1
    return round(1.0 - copied / len(gen_lines), 4)


# ── Combined ────────────────────────────────────────────────────────────────────

def score_generation(source_text: str, generated_text: str, target_author: str) -> dict:
    """Every metric above for one generation, flat, ready to be a CSV row or a
    GRPO reward's component dict."""
    return {
        **repetition_metrics(generated_text),
        **language_metrics(generated_text),
        **structure_fit(generated_text, target_author),
        **style_match(generated_text, target_author),
        **content_preservation(source_text, generated_text),
    }
