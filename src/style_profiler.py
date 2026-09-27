import csv
import re
from collections import defaultdict, Counter
import math
from pathlib import Path

from style_features import (  # noqa: F401 (re-exported for existing importers)
    CONTENT_POS, MATTR_WINDOW, line_final_syllable, line_stats, mattr,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

MIN_POEMS = 5
def main():
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
            s = line_stats(poem)
            author_line_stats[author]['total_tokens'] += s['tokens']
            author_line_stats[author]['total_lines'] += s['lines']
            author_line_stats[author]['total_poems'] += 1
            author_line_stats[author]['total_stanzas'] += s['stanzas']
            author_line_stats[author]['total_stanza_lines'] += s['stanza_lines']

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
    poem_token_map = defaultdict(list)  # (author, poem_id) -> list of pos tags
    for row in pos_rows:
        if row['author'] not in eligible_authors:
            continue
        poem_token_map[(row['author'], row['poem_id'])].append(row['pos'])

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


if __name__ == '__main__':
    main()
