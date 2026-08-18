import csv
import re
from collections import defaultdict, Counter
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

MIN_POEMS = 5
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


# Load raw songs for line-level features
songs = []
with open(DATA / 'stripped_songs.csv') as f:
    for row in csv.DictReader(f):
        songs.append(row)

# Count poems per author; filter to MIN_POEMS
author_poems = defaultdict(list)
for row in songs:
    author_poems[row['author']].append(row['song_text'])

eligible_authors = {a for a, poems in author_poems.items() if len(poems) >= MIN_POEMS}

# Line-level features: avg tokens per line, avg lines per poem, and
# stanza-level structure (song_text stanzas are blank-line delimited — the
# same convention llm_probe.load_stanza() splits on).
author_line_stats = defaultdict(lambda: {
    'total_tokens': 0, 'total_lines': 0, 'total_poems': 0,
    'total_stanzas': 0, 'total_stanza_lines': 0,
})
for author, poems in author_poems.items():
    if author not in eligible_authors:
        continue
    for poem in poems:
        lines = [l for l in poem.splitlines() if l.strip()]
        tokens_per_line = [len(l.split()) for l in lines]
        author_line_stats[author]['total_tokens'] += sum(tokens_per_line)
        author_line_stats[author]['total_lines'] += len(lines)
        author_line_stats[author]['total_poems'] += 1

        stanzas = [s for s in poem.strip().split('\n\n') if s.strip()]
        author_line_stats[author]['total_stanzas'] += len(stanzas)
        author_line_stats[author]['total_stanza_lines'] += sum(
            len([l for l in s.splitlines() if l.strip()]) for s in stanzas
        )

# Load POS-tagged tokens
pos_rows = []
with open(DATA / 'pos_tagged.csv') as f:
    pos_rows = list(csv.DictReader(f))

# Group tokens per author
author_tokens = defaultdict(list)   # author -> list of (lemma, pos)
author_words = defaultdict(list)    # author -> list of surface words (for TTR)
poem_pos_seqs = defaultdict(list)   # author -> list of pos sequences per poem

for row in pos_rows:
    if row['author'] not in eligible_authors:
        continue
    author_tokens[row['author']].append((row['lemma'], row['pos']))
    author_words[row['author']].append(row['word'])

# POS sequences per poem (for bigrams)
poem_token_map = defaultdict(list)  # (author, song_title) -> list of pos tags
for row in pos_rows:
    if row['author'] not in eligible_authors:
        continue
    poem_token_map[(row['author'], row['song_title'])].append(row['pos'])

# Build per-author profiles
profiles = []
for author in sorted(eligible_authors):
    tokens = author_tokens[author]
    words = author_words[author]
    stats = author_line_stats[author]

    total = len(tokens)
    if total == 0:
        continue

    # POS distribution
    pos_counts = Counter(pos for _, pos in tokens)
    pos_dist = {pos: round(pos_counts.get(pos, 0) / total, 4) for pos in CONTENT_POS}

    # Type-token ratio (lexical richness) — use lemmas. Diagnostic only: not
    # length-normalized, so it is not comparable across authors of different
    # corpus sizes (see MATTR_WINDOW comment above) — use 'mattr' for that.
    lemmas = [l for l, _ in tokens]
    ttr = round(len(set(lemmas)) / len(lemmas), 4) if lemmas else 0
    mattr_score = mattr(lemmas)

    # Avg tokens per line, avg lines per poem
    n_poems = stats['total_poems']
    n_lines = stats['total_lines']
    n_tokens_raw = stats['total_tokens']
    avg_tokens_per_line = round(n_tokens_raw / n_lines, 2) if n_lines else 0
    avg_lines_per_poem = round(n_lines / n_poems, 2) if n_poems else 0

    # Stanza-level structure
    n_stanzas = stats['total_stanzas']
    n_stanza_lines = stats['total_stanza_lines']
    avg_stanzas_per_poem = round(n_stanzas / n_poems, 2) if n_poems else 0
    avg_lines_per_stanza = round(n_stanza_lines / n_stanzas, 2) if n_stanzas else 0

    # Top 5 POS bigrams (syntactic fingerprint)
    bigram_counter = Counter()
    for (a, title), pos_seq in poem_token_map.items():
        if a != author:
            continue
        for i in range(len(pos_seq) - 1):
            bigram_counter[(pos_seq[i], pos_seq[i + 1])] += 1
    top_bigrams = [f"{b1}>{b2}" for (b1, b2), _ in bigram_counter.most_common(5)]

    # Rhyme approximation: rime (last vowel onward) of the last real word per line
    rhyme_endings = Counter()
    for poem in author_poems[author]:
        for line in poem.splitlines():
            ending = line_final_syllable(line)
            if ending:
                rhyme_endings[ending] += 1
    top_rhymes = [e for e, _ in rhyme_endings.most_common(5)]

    profiles.append({
        'author': author,
        'num_poems': n_poems,
        'num_tokens': total,
        'pct_NOUN': pos_dist['NOUN'],
        'pct_VERB': pos_dist['VERB'],
        'pct_ADJ': pos_dist['ADJ'],
        'pct_ADV': pos_dist['ADV'],
        'pct_PROPN': pos_dist['PROPN'],
        'type_token_ratio': ttr,
        'mattr': mattr_score,
        'avg_tokens_per_line': avg_tokens_per_line,
        'avg_lines_per_poem': avg_lines_per_poem,
        'avg_stanzas_per_poem': avg_stanzas_per_poem,
        'avg_lines_per_stanza': avg_lines_per_stanza,
        'top_pos_bigrams': '|'.join(top_bigrams),
        'top_rhyme_endings': '|'.join(top_rhymes),
    })

fields = list(profiles[0].keys())
with open(DATA / 'author_style_profiles.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(profiles)

print(f"Style profiles written for {len(profiles)} authors → author_style_profiles.csv")

# Print a quick sample
print(f"\nSample profiles:")
for p in profiles[:3]:
    print(f"  {p['author']}: NOUN={p['pct_NOUN']}, VERB={p['pct_VERB']}, "
          f"TTR={p['type_token_ratio']} (diagnostic only), MATTR={p['mattr']}, "
          f"avg_line_len={p['avg_tokens_per_line']}, "
          f"stanzas/poem={p['avg_stanzas_per_poem']}, lines/stanza={p['avg_lines_per_stanza']}, "
          f"bigrams={p['top_pos_bigrams']}")
