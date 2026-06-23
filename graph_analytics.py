import csv
import networkx as nx
import community as community_louvain

G = nx.read_gexf('skg_enriched.gexf')

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
# Run on the full graph (undirected, already is)
partition = community_louvain.best_partition(G, weight='weight', random_state=42)

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

with open('graph_analytics.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved graph_analytics.csv ({len(rows)} nodes)")
