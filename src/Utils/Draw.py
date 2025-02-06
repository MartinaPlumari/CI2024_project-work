# ------------------------------------------------------
# This code is licensed under the MIT License.
# Copyright (c) 2025 Martina Plumari, Daniel Bologna.
# Developed for the course "Computational Intelligence" 
# at Politecnico di Torino.
# ------------------------------------------------------

import networkx as nx
import matplotlib.pyplot as plt
from tree.node import Node

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, xcenter=0.5, pos=None, level=0):
	"""Funzione per posizionare i nodi in modo gerarchico (top-down)."""
	if pos is None:
		pos = {root: (xcenter, 1 - level * vert_gap)}
	else:
		pos[root] = (xcenter, 1 - level * vert_gap)

	children = list(G.successors(root))
	if len(children) != 0:
		dx = width / max(1, len(children))  # Spazio tra i figli
		nextx = xcenter - width / 2 - dx / 2
		for child in children:
			nextx += dx
			pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, xcenter=nextx, pos=pos, level=level + 1)

	return pos

# # Definizione dell'albero (padre -> figli)
# edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]

# # Etichette dei nodi
# labels = {1: "Root", 2: "A", 3: "B", 4: "C", 5: "D", 6: "E", 7: "F"}

def compute_edges(node : Node, counter=[2], node_id_map={}, edges=[], labels={}, parent_id=None) -> tuple[list, object]:
	# Assegna un ID univoco al nodo
	node_id = counter[0]
	node_id_map[node] = node_id
	counter[0] += 1
	    
	labels[node_id] = node.short_name
	if parent_id is not None:
		edges.append((parent_id, node_id))
	
	for child in node._successors:
		compute_edges(child, counter, node_id_map, edges, labels, node_id)

	return edges, labels


def draw_tree(root : Node) -> None:
	
	(edges, labels) = compute_edges(root, parent_id=1)

	print(edges)
	print(labels)
	
	# Creazione del grafo
	G = nx.DiGraph()
	G.add_edges_from(edges)

	# Calcolo della posizione dei nodi
	pos = hierarchy_pos(G, root=1)

	# Disegno dell'albero
	plt.figure(figsize=(8, 6))
	nx.draw(G, pos, with_labels=False, node_color="lightblue", edge_color="black", node_size=1200, arrows=False)
	nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color="black")

	plt.show()
