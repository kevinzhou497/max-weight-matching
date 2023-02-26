from blossom import * 
import networkx as nx
from itertools import combinations, repeat

G = nx.Graph()

G.add_edge("a", "b", weight=0.6)
G.add_edge("a", "c", weight=0.2)
G.add_edge("c", "d", weight=0.1)
G.add_edge("c", "e", weight=0.7)
G.add_edge("c", "f", weight=0.9)
G.add_edge("a", "d", weight=0.3)
G.add_edge("a", "e", weight=0.8)

print(blossom_algorithm(G))

G2 = nx.Graph()
edges = [(1, 2, 6), (1, 3, 2), (2, 3, 1), (2, 4, 7), (3, 5, 9), (4, 5, 3)]
G2.add_weighted_edges_from(edges)
print(blossom_algorithm(G2))
