import csv
from collections import defaultdict, Counter
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'

MIN_COOCCURRENCE = 2  # min times two words must co-occur across poems to get an edge
MIN_WORD_FREQ = 2     # min times a (lemma, pos) must appear to become a node

# Load TF-IDF results
tfidf = defaultdict(dict)  # author -> {surface_word: score}
with open(DATA / 'tfidf_results.csv') as f:
    for row in csv.DictReader(f):
        tfidf[row['author']][row['word']] = float(row['tfidf_score'])

# Load POS-tagged tokens
pos_rows = []
with open(DATA / 'pos_tagged.csv') as f:
    pos_rows = list(csv.DictReader(f))

# Build word frequency and best (lemma, pos) mapping per surface word per author
# surface_word -> Counter of (lemma, pos) pairs
word_to_lemmapos = defaultdict(Counter)
for r in pos_rows:
    word_to_lemmapos[r['word']][(r['lemma'], r['pos'])] += 1

# (lemma, pos) global frequency — for MIN_WORD_FREQ filter
lemmapos_freq = Counter((r['lemma'], r['pos']) for r in pos_rows)

# Build co-occurrence: per poem, collect all (lemma, pos) pairs, then pair them up
poem_tokens = defaultdict(set)  # (author, song_title) -> set of (lemma, pos)
for r in pos_rows:
    key = (r['author'], r['song_title'])
    poem_tokens[key].add((r['lemma'], r['pos']))

cooccurrence = Counter()
for tokens in poem_tokens.values():
    tokens = sorted(tokens)
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            cooccurrence[(tokens[i], tokens[j])] += 1

POS_COLORS = {
    'NOUN':  (15, 155, 142),
    'VERB':  (245, 166, 35),
    'ADJ':   (123, 104, 238),
    'ADV':   (80, 200, 120),
    'PROPN': (255, 107, 157),
}
AUTHOR_COLOR = (233, 69, 96)

# Build graph
G = nx.Graph()

# Author nodes
author_poem_count = Counter(r['author'] for r in pos_rows)
max_poems = max(author_poem_count.values())
for author, count in author_poem_count.items():
    r, g, b = AUTHOR_COLOR
    G.add_node(author, node_type='author', poem_count=count,
               viz={'color': {'r': r, 'g': g, 'b': b, 'a': 0},
                    'size': float(10 + 30 * (count / max_poems))})

# Word nodes + author->word edges (from TF-IDF)
# Track max TF-IDF score per word node for sizing
word_max_score = defaultdict(float)
word_node_data = {}

for author, words in tfidf.items():
    for surface_word, score in words.items():
        if surface_word not in word_to_lemmapos:
            continue
        lemma, pos = word_to_lemmapos[surface_word].most_common(1)[0][0]
        if lemmapos_freq[(lemma, pos)] < MIN_WORD_FREQ:
            continue
        node_id = f'{lemma}_{pos}'
        word_max_score[node_id] = max(word_max_score[node_id], score)
        word_node_data[node_id] = (lemma, pos)

max_score = max(word_max_score.values()) if word_max_score else 1
for node_id, (lemma, pos) in word_node_data.items():
    r, g, b = POS_COLORS.get(pos, (136, 136, 136))
    G.add_node(node_id, node_type='word', lemma=lemma, pos=pos,
               viz={'color': {'r': r, 'g': g, 'b': b, 'a': 0},
                    'size': float(3 + 12 * (word_max_score[node_id] / max_score))})

for author, words in tfidf.items():
    for surface_word, score in words.items():
        if surface_word not in word_to_lemmapos:
            continue
        lemma, pos = word_to_lemmapos[surface_word].most_common(1)[0][0]
        node_id = f'{lemma}_{pos}'
        if G.has_node(node_id):
            G.add_edge(author, node_id, edge_type='distinctive', weight=score)

# Word->word co-occurrence edges
for (lp1, lp2), count in cooccurrence.items():
    if count < MIN_COOCCURRENCE:
        continue
    n1 = f'{lp1[0]}_{lp1[1]}'
    n2 = f'{lp2[0]}_{lp2[1]}'
    if G.has_node(n1) and G.has_node(n2):
        G.add_edge(n1, n2, edge_type='cooccurrence', weight=count)

nx.write_gexf(G, MODELS / 'skg.gexf')

print(f"SKG built:")
print(f"  Nodes: {G.number_of_nodes()} ({sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'author')} authors, {sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'word')} words)")
print(f"  Edges: {G.number_of_edges()} ({sum(1 for *_, d in G.edges(data=True) if d.get('edge_type') == 'distinctive')} author→word, {sum(1 for *_, d in G.edges(data=True) if d.get('edge_type') == 'cooccurrence')} word↔word)")
print(f"Saved to models/skg.gexf")
print("\nIn Gephi: File → Open models/skg.gexf, then run Layout → ForceAtlas2")

# Sanity check: top distinctive words for one author
sample_author = sorted(tfidf.keys())[0]
neighbors = [(d['lemma'], d['pos'], G[sample_author][n]['weight'])
             for n, d in G.nodes(data=True)
             if G.has_edge(sample_author, n) and d.get('node_type') == 'word']
neighbors.sort(key=lambda x: -x[2])
print(f"\nTop words for '{sample_author}':")
for lemma, pos, w in neighbors[:10]:
    print(f"  {lemma} ({pos}) — {w:.4f}")
