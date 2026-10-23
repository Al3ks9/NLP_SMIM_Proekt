import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'

MIN_POEMS = 5

# Load POS-tagged tokens
pos_rows = []
with open(DATA / 'pos_tagged.csv') as f:
    pos_rows = list(csv.DictReader(f))

# Load TF-IDF scores
tfidf = defaultdict(dict)  # author -> {word: score}
with open(DATA / 'tfidf_results.csv') as f:
    for row in csv.DictReader(f):
        tfidf[row['author']][row['word']] = float(row['tfidf_score'])

# Filter to eligible authors
author_poem_counts = Counter((r['author'], r['poem_id']) for r in pos_rows)
author_counts = Counter(r['author'] for r in pos_rows)
eligible_authors = {
    author for author in author_counts
    if len({r['poem_id'] for r in pos_rows if r['author'] == author}) >= MIN_POEMS
}

# --- Build per-author vocabulary: author -> POS -> [(lemma, freq)] ---
author_pos_vocab = defaultdict(lambda: defaultdict(Counter))
for r in pos_rows:
    if r['author'] not in eligible_authors:
        continue
    author_pos_vocab[r['author']][r['pos']][r['lemma']] += 1

# Serialize to plain dict for JSON
vocab_data = {}
for author, pos_map in author_pos_vocab.items():
    vocab_data[author] = {}
    for pos, lemma_counts in pos_map.items():
        # Sort by frequency descending
        vocab_data[author][pos] = sorted(
            [(lemma, cnt) for lemma, cnt in lemma_counts.items()],
            key=lambda x: -x[1]
        )

with open(MODELS / 'author_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)

print(f"Saved author_vocab.json ({len(vocab_data)} authors)")

# --- Build POS bigram transition matrices per author ---
# Per poem, collect POS sequence, then count bigrams
pos_transitions = {}  # author -> {pos: {next_pos: count}}

poem_pos_seqs = defaultdict(list)
for r in pos_rows:
    if r['author'] not in eligible_authors:
        continue
    poem_pos_seqs[(r['author'], r['poem_id'])].append(r['pos'])

author_bigrams = defaultdict(lambda: defaultdict(Counter))
for (author, _), seq in poem_pos_seqs.items():
    for i in range(len(seq) - 1):
        author_bigrams[author][seq[i]][seq[i + 1]] += 1

# Convert counts to transition probabilities
for author, from_pos_map in author_bigrams.items():
    pos_transitions[author] = {}
    for from_pos, to_counts in from_pos_map.items():
        total = sum(to_counts.values())
        pos_transitions[author][from_pos] = {
            to_pos: round(cnt / total, 4)
            for to_pos, cnt in to_counts.most_common()
        }

with open(MODELS / 'pos_transitions.json', 'w', encoding='utf-8') as f:
    json.dump(pos_transitions, f, ensure_ascii=False, indent=2)

print(f"Saved pos_transitions.json ({len(pos_transitions)} authors)")

# --- Load existing SKG and enrich it ---
G = nx.read_gexf(MODELS / 'skg.gexf')

# Add vocab and transition data as JSON string attributes on author nodes
for node_id, data in G.nodes(data=True):
    if data.get('node_type') != 'author':
        continue
    author = node_id
    if author in vocab_data:
        G.nodes[node_id]['vocab_json'] = json.dumps(
            vocab_data[author], ensure_ascii=False
        )
    if author in pos_transitions:
        G.nodes[node_id]['transitions_json'] = json.dumps(
            pos_transitions[author], ensure_ascii=False
        )

# --- Add author-similarity edges based on vocabulary overlap ---
# Jaccard over full lemma vocabulary (all POS combined) from pos_tagged
authors_in_graph = [
    n for n, d in G.nodes(data=True) if d.get('node_type') == 'author'
]
author_full_vocab = defaultdict(set)
for r in pos_rows:
    if r['author'] in eligible_authors:
        author_full_vocab[r['author']].add(r['lemma'])

author_top_words = {a: author_full_vocab[a] for a in authors_in_graph if a in author_full_vocab}

SIMILARITY_THRESHOLD = 0.15  # Jaccard over full lemma set
added = 0
author_list = sorted(author_top_words.keys())
for i in range(len(author_list)):
    for j in range(i + 1, len(author_list)):
        a1, a2 = author_list[i], author_list[j]
        w1, w2 = author_top_words[a1], author_top_words[a2]
        union = w1 | w2
        if not union:
            continue
        jaccard = len(w1 & w2) / len(union)
        if jaccard >= SIMILARITY_THRESHOLD:
            G.add_edge(a1, a2, edge_type='author_similarity', weight=round(jaccard, 4))
            added += 1

print(f"Added {added} author-similarity edges (Jaccard ≥ {SIMILARITY_THRESHOLD})")

nx.write_gexf(G, MODELS / 'skg_enriched.gexf')
print(f"Saved skg_enriched.gexf — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Quick sanity check
sample = author_list[0]
print(f"\nSample vocab for '{sample}':")
for pos, words in list(vocab_data[sample].items())[:3]:
    print(f"  {pos}: {[w for w, _ in words[:5]]}")
print(f"\nSample POS transitions for '{sample}':")
for from_pos, targets in list(pos_transitions[sample].items())[:3]:
    top = sorted(targets.items(), key=lambda x: -x[1])[:3]
    print(f"  {from_pos} → {top}")
