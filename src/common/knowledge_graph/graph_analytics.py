import csv
from pathlib import Path
import networkx as nx
import community as community_louvain

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'

G = nx.read_gexf(MODELS / 'skg_final.gexf')

# --- PageRank over word nodes ---
pr = nx.pagerank(G, weight='weight')
word_pr = sorted(
    [(n, pr[n], d.get('lemma', n), d.get('pos', ''))
     for n, d in G.nodes(data=True) if d.get('node_type') == 'word'],
    key=lambda x: -x[1]
)

print("Top 15 most central words (PageRank):")
for node_id, score, lemma, pos in word_pr[:15]:
    print(f"  {lemma:20s} ({pos:5s})  PR={score:.5f}")

# --- Weighted degree centrality per author ---
author_degree = sorted(
    [(n, G.degree(n, weight='weight'), d.get('node_type'))
     for n, d in G.nodes(data=True) if d.get('node_type') == 'author'],
    key=lambda x: -x[1]
)

print("\nAuthor weighted degree (breadth of distinctive vocabulary in graph):")
for author, deg, _ in author_degree:
    print(f"  {author:35s}  weighted_degree={deg:.2f}")

# --- Community detection (Louvain) ---
# Run on word-mediated edges only (cooccurrence, semantic_similarity,
# distinctive) — author_similarity edges are excluded from the partitioning
# graph. Those edges are a direct author<->author Jaccard-vocabulary-overlap
# signal (enrich_skg.py, threshold 0.15), not evidence mediated by anything
# either author actually wrote, and leaving them in let Louvain cluster
# authors together on vocabulary overlap alone with no word-level support
# behind it: 13 of 24 authors landed in one "community" built on 2 surviving
# word nodes, purely because those authors' weighted degree happened to be
# 40-70% author_similarity edges versus 86-100% word-mediated for everyone
# else. author_similarity is still a real signal — it's just used directly
# (not through this graph) as an auxiliary/fallback source in
# candidate_selection.author_similarity() for tier_cross_author_similar,
# not as something that should define a community on its own.
#
# PageRank and weighted_degree above are intentionally left on the full
# graph G — author_similarity edges are legitimate there (centrality and
# degree are not claims about shared vocabulary the way a community is).
G_community = G.copy()
G_community.remove_edges_from(
    (u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'author_similarity'
)
partition = community_louvain.best_partition(G_community, weight='weight', random_state=42)

# Group by community
communities = {}
for node, comm_id in partition.items():
    communities.setdefault(comm_id, []).append(node)

# Print communities that contain at least one author
print("\nStyle communities (Louvain) — showing author members per community:")
author_nodes = {n for n, d in G.nodes(data=True) if d.get('node_type') == 'author'}
for comm_id in sorted(communities):
    authors_in_comm = [n for n in communities[comm_id] if n in author_nodes]
    if authors_in_comm:
        word_count = len(communities[comm_id]) - len(authors_in_comm)
        print(f"\n  Community {comm_id} ({len(communities[comm_id])} nodes, {word_count} words):")
        for a in authors_in_comm:
            print(f"    - {a}")

# --- Save node-level metrics to CSV ---
rows = []
for node_id, data in G.nodes(data=True):
    rows.append({
        'node_id': node_id,
        'node_type': data.get('node_type', ''),
        'lemma': data.get('lemma', ''),
        'pos': data.get('pos', ''),
        'pagerank': round(pr[node_id], 6),
        'weighted_degree': round(G.degree(node_id, weight='weight'), 4),
        'community': partition[node_id],
    })

with open(DATA / 'graph_analytics.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved graph_analytics.csv ({len(rows)} nodes)")
