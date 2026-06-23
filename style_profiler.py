import csv
from collections import defaultdict, Counter
import math

MIN_POEMS = 5
CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

# Load raw songs for line-level features
songs = []
with open('stripped_songs.csv') as f:
    for row in csv.DictReader(f):
        songs.append(row)

# Count poems per author; filter to MIN_POEMS
author_poems = defaultdict(list)
for row in songs:
    author_poems[row['author']].append(row['song_text'])

eligible_authors = {a for a, poems in author_poems.items() if len(poems) >= MIN_POEMS}

# Line-level features: avg tokens per line, avg lines per poem
author_line_stats = defaultdict(lambda: {'total_tokens': 0, 'total_lines': 0, 'total_poems': 0})
for author, poems in author_poems.items():
    if author not in eligible_authors:
        continue
    for poem in poems:
        lines = [l for l in poem.splitlines() if l.strip()]
        tokens_per_line = [len(l.split()) for l in lines]
        author_line_stats[author]['total_tokens'] += sum(tokens_per_line)
        author_line_stats[author]['total_lines'] += len(lines)
        author_line_stats[author]['total_poems'] += 1

# Load POS-tagged tokens
pos_rows = []
with open('pos_tagged.csv') as f:
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

    # Type-token ratio (lexical richness) — use lemmas
    lemmas = [l for l, _ in tokens]
    ttr = round(len(set(lemmas)) / len(lemmas), 4) if lemmas else 0

    # Avg tokens per line, avg lines per poem
    n_poems = stats['total_poems']
    n_lines = stats['total_lines']
    n_tokens_raw = stats['total_tokens']
    avg_tokens_per_line = round(n_tokens_raw / n_lines, 2) if n_lines else 0
    avg_lines_per_poem = round(n_lines / n_poems, 2) if n_poems else 0

    # Top 5 POS bigrams (syntactic fingerprint)
    bigram_counter = Counter()
    for (a, title), pos_seq in poem_token_map.items():
        if a != author:
            continue
        for i in range(len(pos_seq) - 1):
            bigram_counter[(pos_seq[i], pos_seq[i + 1])] += 1
    top_bigrams = [f"{b1}>{b2}" for (b1, b2), _ in bigram_counter.most_common(5)]

    # Rhyme approximation: last 3 chars of last word per line
    rhyme_endings = Counter()
    for poems_list in [author_poems[author]]:
        for poem in poems_list:
            for line in poem.splitlines():
                words_in_line = line.strip().split()
                if words_in_line:
                    rhyme_endings[words_in_line[-1][-3:]] += 1
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
        'avg_tokens_per_line': avg_tokens_per_line,
        'avg_lines_per_poem': avg_lines_per_poem,
        'top_pos_bigrams': '|'.join(top_bigrams),
        'top_rhyme_endings': '|'.join(top_rhymes),
    })

fields = list(profiles[0].keys())
with open('author_style_profiles.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(profiles)

print(f"Style profiles written for {len(profiles)} authors → author_style_profiles.csv")

# Print a quick sample
print(f"\nSample profiles:")
for p in profiles[:3]:
    print(f"  {p['author']}: NOUN={p['pct_NOUN']}, VERB={p['pct_VERB']}, "
          f"TTR={p['type_token_ratio']}, avg_line_len={p['avg_tokens_per_line']}, "
          f"bigrams={p['top_pos_bigrams']}")
