"""Spectral algorithm for community detection."""


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
import pdb


n_mcmc_iter = 500


def swap_edges(g, u1, v):
    """Swap (u1, v) edge with a random matching edge."""

    deg_u1 = g.degree(u1)
    same_deg_nodes = [n for n, d in g.degree().iteritems()
                      if d == deg_u1 and n != u1 and n not in g.neighbors(u1)]
    if len(same_deg_nodes) > 0:
        u2 = choice(same_deg_nodes)
        cand = [ng for ng in g.neighbors(u2) if ng not in g.neighbors(u1)]
        if len(cand) > 0:
            w = choice(cand)

            g.add_edge(u1, w)
            g.add_edge(u2, v)
            g.remove_edge(u1, v)
            g.remove_edge(u2, w)


def generate_mcmc(original_graph, n_iter=500):
    """Return MCMC variant of a graph."""

    new_graph = original_graph.copy()

    for i in xrange(n_iter):
        candidates = [c for c in new_graph.nodes()
                    if new_graph.degree(c) > 0]
        c1 = choice(candidates)
        c2 = choice(new_graph.neighbors(c1))
        swap_edges(new_graph, c1, c2)

    return new_graph


def split_communities(g):
    """Return two communities associated to g."""

    a = nx.adjacency_matrix(g)
    inv_sqrt_d = np.diag(1 / np.sqrt(nx.degree(g).values()))
    l = inv_sqrt_d.dot(a.dot(inv_sqrt_d))
    eig_val, eig_vec = np.linalg.eig(l)
    s = np.ravel((eig_vec[:, 1] > 0))
    return s


def draw_communities(g, s):
    """Plot communities and stats about (g, s)."""

    nodes = nx.nodes(g)
    edges = nx.edges(g)
    nodes_1 = [nodes[i] for i in xrange(len(nodes)) if s[i]]
    nodes_2 = [nodes[i] for i in xrange(len(nodes)) if not s[i]]
    inner_1 = [e for e in edges if e[0] in nodes_1 and e[1] in nodes_1]
    inner_2 = [e for e in edges if e[0] in nodes_2 and e[1] in nodes_2]
    outter = [e for e in edges if e not in inner_1 and e not in inner_2]
    deg1 = [i + 5 for i in g.degree(nodes_1).values()]
    deg2 = [i + 5 for i in g.degree(nodes_2).values()]

    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g,
                           pos,
                           nodelist=nodes_1,
                           node_color=deg1,
                           node_size=500,
                           cmap=plt.cm.Reds,
                           alpha=.8)
    nx.draw_networkx_nodes(g,
                           pos,
                           nodelist=nodes_2,
                           node_color=deg2,
                           node_size=500,
                           cmap=plt.cm.Greens,
                           alpha=.8)
    #nx.draw_networkx_labels(g,
    #                        pos,
    #                        labels={i: str(i + 1) for i in range(len(nodes))})
    nx.draw_networkx_edges(g,
                           pos,
                           width=1.,
                           alpha=.5)
    nx.draw_networkx_edges(g,
                           pos,
                           edgelist=inner_1,
                           width=4.,
                           alpha=.5,
                           edge_color='r')
    nx.draw_networkx_edges(g,
                           pos,
                           edgelist=inner_2,
                           width=4.,
                           alpha=.5,
                           edge_color='g')
    print "---------------------------------------"
    print "#Nodes (red community)         : {0}".format(len(nodes_1))
    print "#Nodes (green community)       : {0}".format(len(nodes_2))
    print "#Inner edges (red community)   : {0}".format(len(inner_1))
    print "#Inner edges (green community) : {0}".format(len(inner_2))
    print "#Edges between communities     : {0}".format(len(outter))
    print "---------------------------------------"


g = nx.generators.karate_club_graph()
#plt.subplot(121)
plt.axis('off')
plt.title('Karate Club')
draw_communities(g, split_communities(g))

#test = generate_mcmc(g, n_iter=n_mcmc_iter)
#plt.subplot(122)
#plt.axis('off')
#plt.title('MCMC mix ({0} iterations)'.format(n_mcmc_iter))
#draw_communities(test, split_communities(test))

plt.show()

#g = nx.generators.karate_club_graph()
#plt.axis('off')
#plt.title('Karate Club')
#pos = nx.spring_layout(g)
#deg = g.degree(g.nodes())
#nx.draw_networkx_nodes(g,
#                       pos,
#                       node_list=deg.keys(),
#                       node_size=300,
#                       node_color=deg.values(),
#                       cmap=plt.cm.Blues)
#nx.draw_networkx_edges(g, pos, width=1., alpha=.5, edge_color='k')
#plt.show()
