"""
Per-poem stylometric features, shared by style_profiler.py (author profiles),
build_reward_calibration.py (per-author feature distributions) and
grpo_rewards.py (scoring a generated poem) -- one implementation, so the GRPO
style reward measures exactly what the profiles describe.

style_profiler.py aggregates author-level *totals* (e.g. total NOUN tokens /
total tokens); the reward needs per-poem values, because a single poem is
what gets scored. poem_features() returns those, unrounded.
"""

import math
import re
from collections import Counter

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

_VOWELS = set('аеиоу')
_EDGE_PUNCT_RE = re.compile(r'^[^\w]+|[^\w]+$', re.UNICODE)


def line_final_syllable(line: str):
    """
    The line's rhyme-relevant ending: the last orthographic syllable of the
    last real word — everything from the onset consonant(s) of the final
    vowel nucleus (i.e. right after the second-to-last vowel) through the end
    of the word — after stripping the punctuation that clings to line-final
    words (–|,|.|!... and the like). A single-vowel word has no
    second-to-last vowel to anchor on, so it falls back to last-vowel-onward.

    Walks backward past trailing tokens that are pure punctuation — a bare
    em/en dash used as a caesura marker is common at Macedonian line ends and
    carries no rhyme information — and returns None if the line has no real
    word at all. A "word" with no vowel (a bare number, an initialism) is
    likewise not rhyme-bearing and is skipped the same way.
    """
    for raw in reversed(line.strip().split()):
        word = _EDGE_PUNCT_RE.sub('', raw).lower()
        if word.isalpha() and any(ch in _VOWELS for ch in word):
            vowel_idx = [i for i, ch in enumerate(word) if ch in _VOWELS]
            start = vowel_idx[-2] + 1 if len(vowel_idx) >= 2 else vowel_idx[-1]
            return word[start:]
    return None


# Plain type-token ratio falls mechanically as corpus size grows — every
# author's num_tokens here spans a 37x range (232-8538), and on this corpus
# plain TTR correlates with num_tokens at Pearson r=-0.76 (p<1e-4). MATTR
# (Covington & McFall 2010) averages the TTR of every fixed-size sliding
# window instead of one TTR over the whole text, which is what makes it
# comparable across authors of very different corpus sizes. The window has
# to fit inside the smallest eligible author's token count (232), so 200
# leaves margin while still being long enough to smooth out per-window noise.
MATTR_WINDOW = 200


def mattr(lemmas: list, window: int = MATTR_WINDOW) -> float:
    """
    Moving-Average Type-Token Ratio. Falls back to plain TTR when the corpus
    is shorter than one window (not expected among eligible authors here,
    but kept for safety if MIN_POEMS or the corpus ever shrinks a profile
    below the window).
    """
    n = len(lemmas)
    if n == 0:
        return 0
    if n <= window:
        return round(len(set(lemmas)) / n, 4)

    counts = Counter(lemmas[:window])
    ratios = [len(counts) / window]
    for i in range(window, n):
        outgoing = lemmas[i - window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[lemmas[i]] += 1
        ratios.append(len(counts) / window)
    return round(sum(ratios) / len(ratios), 4)


POS_PCT = ('NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN')
SCALAR_FEATURES = tuple(f'pct_{p}' for p in POS_PCT) + (
    'lexical_diversity', 'avg_tokens_per_line', 'num_lines', 'num_stanzas',
    'avg_lines_per_stanza')
DIST_FEATURES = ('pos_bigrams', 'rhyme_endings')


def line_stats(text: str) -> dict:
    """Whitespace-token, line and blank-line-stanza counts -- the exact
    conventions style_profiler.py's line-level features use."""
    lines = [l for l in text.splitlines() if l.strip()]
    stanzas = [s for s in text.strip().split('\n\n') if s.strip()]
    return {
        'tokens': sum(len(l.split()) for l in lines),
        'lines': len(lines),
        'stanzas': len(stanzas),
        'stanza_lines': sum(len([l for l in s.splitlines() if l.strip()]) for s in stanzas),
    }


def pos_bigrams(pos_seq: list) -> Counter:
    return Counter(f'{a}>{b}' for a, b in zip(pos_seq, pos_seq[1:]))


def rhyme_endings(text: str) -> Counter:
    counts = Counter()
    for line in text.splitlines():
        ending = line_final_syllable(line)
        if ending:
            counts[ending] += 1
    return counts


def poem_features(tokens: list, text: str) -> dict:
    """
    tokens: [(lemma, pos), ...] content tokens of the poem, selected and
    normalised exactly as pos_tagged.csv holds them (tagging.tag_poem for
    generated text). text: the raw poem.
    """
    total = len(tokens)
    pos_counts = Counter(p for _, p in tokens)
    stats = line_stats(text)
    feats = {f'pct_{p}': (pos_counts[p] / total if total else 0.0) for p in POS_PCT}
    feats['lexical_diversity'] = float(mattr([l for l, _ in tokens]))
    feats['avg_tokens_per_line'] = stats['tokens'] / stats['lines'] if stats['lines'] else 0.0
    feats['num_lines'] = float(stats['lines'])
    feats['num_stanzas'] = float(stats['stanzas'])
    feats['avg_lines_per_stanza'] = (stats['stanza_lines'] / stats['stanzas']
                                     if stats['stanzas'] else 0.0)
    feats['pos_bigrams'] = pos_bigrams([p for _, p in tokens])
    feats['rhyme_endings'] = rhyme_endings(text)
    return feats


def normalise(counts) -> dict:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def dist_cosine(counts, dist: dict) -> float:
    """Cosine between a count vector and a (normalised) distribution; 0.0
    when either side is empty."""
    dot = sum(v * dist.get(k, 0.0) for k, v in counts.items())
    na = math.sqrt(sum(v * v for v in counts.values()))
    nb = math.sqrt(sum(v * v for v in dist.values()))
    return dot / (na * nb) if na and nb else 0.0
