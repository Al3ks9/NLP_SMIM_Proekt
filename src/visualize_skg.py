from pathlib import Path
import networkx as nx
from pyvis.network import Network

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'

G = nx.read_gexf(MODELS / 'skg_final.gexf')

net = Network(height='900px', width='100%', bgcolor='#1a1a2e', font_color='white', notebook=False)
net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=150)

for node_id, data in G.nodes(data=True):
    if data.get('node_type') == 'author':
        net.add_node(node_id, label=node_id, color='#e94560', size=25,
                     title=f"Author: {node_id}\nPoems: {data.get('poem_count', '?')}")
    else:
        pos = data.get('pos', '')
        color = {'NOUN': '#0f9b8e', 'VERB': '#f5a623', 'ADJ': '#7b68ee',
                 'ADV': '#50c878', 'PROPN': '#ff6b9d'}.get(pos, '#888888')
        net.add_node(node_id, label=data.get('lemma', node_id), color=color, size=10,
                     title=f"{data.get('lemma')} ({pos})")

for u, v, data in G.edges(data=True):
    if data.get('edge_type') == 'distinctive':
        net.add_edge(u, v, color='rgba(255,255,255,0.4)', width=data.get('weight', 0.1) * 5)
    else:
        net.add_edge(u, v, color='rgba(255,255,255,0.1)', width=1)

net.save_graph(str(ROOT / 'skg_visual.html'))
print("Saved to skg_visual.html — open it in a browser")
