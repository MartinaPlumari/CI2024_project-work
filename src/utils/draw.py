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
	"""Foreach node compute a 2D position starting from the root"""
	if pos is None:
		pos = {root: (xcenter, 1 - level * vert_gap)}
	else:
		pos[root] = (xcenter, 1 - level * vert_gap)

	children = list(G.successors(root))
	if len(children) != 0:
		dx = width / max(1, len(children))
		nextx = xcenter - width / 2 - dx / 2
		for child in children:
			nextx += dx
			pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, xcenter=nextx, pos=pos, level=level + 1)

	return pos

def compute_edges(node : Node, counter=[1], node_id_map={}, edges=[], labels={}, parent_id=None) -> tuple[list, object]:
	"""For each node recursively compute each graph edges and labels."""
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
	# compute graph's edges
	(edges, labels) = compute_edges(root, parent_id=0)
	edges.pop(0)
	
	# Graph creation
	G = nx.DiGraph()
	G.add_edges_from(edges)

	# Compute the 2D positions of each nodes to draw the graph as a tree
	pos = hierarchy_pos(G, root=1)

	# Plot the tree
	plt.close('all') 
	plt.figure(figsize=(8, 6))
	nx.draw(G, pos, with_labels=False, node_color="lightblue", edge_color="black", node_size=1200, arrows=False)
	nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color="black")
	plt.show()