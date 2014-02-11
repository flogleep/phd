"""
Implementation of MCMC graph generation with stopping criterion.

Original article: arxiv.org/pdf/1202.3473.pdf

"""


from random import choice


def swap_edges(g, u1, v):
    """Swap (u1, v) edge with a random matching edge."""

    deg_u1 = g.degree(u1)
    same_deg_nodes = [n for n, d in g.degree().iteritems()
                      if d == deg_u1 and n != u1]
    if len(same_deg_nodes) > 0:
        u2 = choice(same_deg_nodes)
        cand = [ng for ng in g.neighbors(u2) if ng != u1 and ng != v]
        if len(cand) > 0:
            w = choice(cand)

            g.add_edge(u1, u2)
            g.add_edge(v, w)
            g.remove_edge(u1, v)
            g.remove_edge(u2, w)


def generate_mcmc(original_graph, n_iter=500):
    """Return MCMC variant of a graph."""

    new_graph = original_graph.copy()
    candidates = [c for c in new_graph.nodes()
                  if new_graph.degree(c) > 0]

    for i in xrange(n_iter):
        c1 = choice(candidates)
        c2 = choice(new_graph.neighbors(c1))
        swap_edges(new_graph, c1, c2)

    return new_graph
