import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pdb
import random as rnd

g = nx.Graph()
g.add_nodes_from(range(8))
g.add_edges_from([(0,1), (0,3),
                  (1,2), (1,3),
                  (2,3),
                  (3,4),
                  (4,5), (4,6), (4,7),
                  (5,6),
                  (6,7)])
l = nx.spring_layout(g)
plt.subplot(121)
nx.draw(g, pos=l, node_size=500, with_labels=False)
nx.draw_networkx_edges(g, pos=l, alpga=.75, width=5)
plt.title('Graphe d\'origine')
plt.subplot(122)
nx.draw_networkx_edges(g, pos=l, alpha=.15, width=5)
g.remove_edge(3, 4)
nx.draw_networkx_edges(g, pos=l, alpga=.75, width=5)
nx.draw(g, pos=l, node_size=500, with_labels=False)
plt.title('Graphe reconstruit')
plt.show()
