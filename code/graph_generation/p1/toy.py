import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pdb
import random as rnd


theta = .05
alphas = .01
betas = .01
rho = 0

g = nx.DiGraph()
g.add_nodes_from(range(10))

for i in xrange(1, 9):
    for j in xrange(i + 1, 10):
        p10 = alphas + betas + theta
        p01 = alphas + betas + theta
        if rnd.random() < p10:
            g.add_edge(i, j)
        if rnd.random() < p01:
            g.add_edge(j, i)

beta0s =[.01, .1, .4, .8]
alpha0 = .1
cur = 0

for beta0 in beta0s:
    gc = g.copy()
    cur += 1
    for i in xrange(1, 10):
        p10 = alpha0 + betas + theta
        p01 = alphas + beta0 + theta
        if rnd.random() < p10:
            gc.add_edge(0, i)
        if rnd.random() < p01:
            gc.add_edge(i, 0)


    l = nx.circular_layout(gc)
    l[0] = [.5, .5]
    plt.subplot(2, 2, cur)
    nx.draw(gc, pos=l)
    plt.title('Popularite : {0}'.format(beta0))
plt.show()
