"""
One-time script: compute multilingual embeddings for all corpus lemmas.

Embeds ALL unique lemmas from pos_tagged.csv (not just SKG word nodes) so that
semantic guidance works for folk/rare words that never made it into the SKG.

Outputs:
  word_embeddings.json  — lemma → embedding vector (for style_transfer.py)
  skg_final.gexf        — SKG with added semantic_similarity edges between
                          same-POS word nodes whose cosine similarity ≥ threshold

Run once: uv run python add_embeddings.py
"""

import csv
import json
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

SIMILARITY_THRESHOLD = 0.65

# --- Collect ALL unique lemmas from the full corpus ---
print("Collecting lemmas from pos_tagged.csv...", flush=True)
all_lemmas: set[str] = set()
with open('pos_tagged.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        all_lemmas.add(row['lemma'])

lemma_list = sorted(all_lemmas)
print(f"Found {len(lemma_list)} unique lemmas in corpus")

# --- Compute embeddings for all corpus lemmas ---
print("Loading model...", flush=True)
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print(f"Computing embeddings...", flush=True)
all_embeddings = model.encode(
    lemma_list, show_progress_bar=True, normalize_embeddings=True, batch_size=256
)

lemma_to_emb = {lemma: all_embeddings[i].tolist() for i, lemma in enumerate(lemma_list)}
with open('word_embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(lemma_to_emb, f, ensure_ascii=False)
print(f"Saved word_embeddings.json  ({len(lemma_to_emb)} entries)")

# --- Add semantic similarity edges to the SKG (SKG word nodes only) ---
print("Loading graph...", flush=True)
G = nx.read_gexf('skg_enriched.gexf')

word_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('node_type') == 'word']
node_ids  = [n for n, d in word_nodes]
node_pos  = {n: d.get('pos', '') for n, d in word_nodes}

# Pull embeddings for SKG word nodes from the already-computed dict
skg_lemmas     = [d.get('lemma', n) for n, d in word_nodes]
skg_embeddings = np.array([lemma_to_emb[l] for l in skg_lemmas])

print("Adding semantic similarity edges to graph...", flush=True)
sim_matrix = skg_embeddings @ skg_embeddings.T  # shape (N_skg, N_skg)

added = 0
for i in range(len(node_ids)):
    for j in range(i + 1, len(node_ids)):
        if node_pos[node_ids[i]] != node_pos[node_ids[j]]:
            continue
        sim = float(sim_matrix[i, j])
        if sim >= SIMILARITY_THRESHOLD:
            G.add_edge(node_ids[i], node_ids[j],
                       edge_type='semantic_similarity',
                       weight=round(sim, 4))
            added += 1

print(f"Added {added} semantic similarity edges (cosine ≥ {SIMILARITY_THRESHOLD}, same POS)")
nx.write_gexf(G, 'skg_final.gexf')
print(f"Saved skg_final.gexf — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
