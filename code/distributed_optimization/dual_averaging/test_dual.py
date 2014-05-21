"""Testing module for dual averaging algorithm."""

import dual_averaging as da
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import math


n = 2 * 1
K = 2
mu_1 = np.array([0., 5.])
mu_2 = np.array([0., -5.])
sigma_1 = np.array([[1., 0.],
                    [0., 1.]])
sigma_2 = sigma_1.copy()

points_1 = np.random.multivariate_normal(mu_1, sigma_1, n / 2)
points_2 = np.random.multivariate_normal(mu_2, sigma_2, n / 2)
points = np.vstack([points_1, points_2])

d = np.zeros((n, n))
#for i in xrange(n):
#    for j in xrange(n):
#        d[i, j] = np.sqrt((points[i] - points[j]).dot(points[i] - points[j]))

d[0, 0] = -1
d[0, 1] = 1
d[1, 0] = 1
d[1, 1] = -1

#plt.scatter(points[:, 0], points[:, 1])
#plt.show()

def generic_f(x, d, i, n):
    return (d[i].dot(x).dot(x[i])) / n

def generic_grad(x, d, i, n):
    return np.atleast_2d(d[i]).transpose().dot(np.atleast_2d(x[i])) / n

x0 = np.zeros((n, K))
x0[:, 0] = 1
fs = lambda x, i: generic_f(x, d, i, n)
grads_f = lambda x, i: generic_grad(x, d, i, n)

alphas = [1. / math.sqrt(t) for t in xrange(1, 501)]
prox = lambda z, alpha: np.exp(-alpha * z)
p = np.ones((n, n)) / n

def glob_f(x):
    res = 0.
    for i in xrange(n):
        res += fs(x, i)
    return res / n

def glob_grad(x):
    res = grads_f(x, 0)
    for i in xrange(1, n):
        res += grads_f(x, i)
    return res / n

av_xs = da.dual_averaging(x0, glob_f, glob_grad, prox, alphas)
#av_xs = da.distributed_dual_averaging(x0, fs, grads_f, prox, alphas, p)
pdb.set_trace()
