import dual_averaging as da
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math

n = 2 * 2

x = np.vstack([.5 * np.random.randn(n / 2, 2) + np.array([3., 3.]),
               .5 * np.random.randn(n / 2, 2) + np.array([-3., -3.])
               ])


def dist_relaxed_clustering(i, a, d):
    """Return value of f_i."""

    return a[i].dot(a.dot(d[i])) / n

def dist_grad_relaxed(i, a, d):
    """Return grad of f_i."""

    g = np.zeros(a.shape)
    g[i] = a.dot(d[i]) / n
    g += np.atleast_2d(a[i]).transpose().dot(np.atleast_2d(d[i])) / n

    return g

def relaxed_clustering(a, d):
    """Return value of f."""

    s = 0
    for i in xrange(n):
        s += dist_relaxed_clustering(i, a, d)
    return s / n

def grad_relaxed(a, d):
    """Return gradient of f."""

    g = np.zeros(a.shape)
    for i in xrange(n):
        g += dist_grad_relaxed(i, a, d)
    return g / n

f = lambda a: relaxed_clustering(u, x, spr, w)
fs = lambda i, a: dist_relaxed_clustering(i, u, x, spr, w)
grad_f = lambda u: grad_relaxed(u, x, spr, w)
grads_f = lambda i, u: dist_grad_relaxed(i, u, x, spr, w)
prox = lambda z, a: -a * z
alphas = [1. / math.sqrt(t) for t in xrange(1, 1000)]
u = x.copy()

new_u = da.dual_averaging(u, f, grad_f, prox, alphas)
plt.scatter(x[:(n / 2),0], x[:(n / 2),1], marker='+', color='b')
plt.scatter(x[(n / 2):,0], x[(n / 2):,1], marker='+', color='r')
plt.scatter(new_u[:(n / 2),0], new_u[:(n / 2),1], marker='o', color='b')
plt.scatter(new_u[(n / 2):,0], new_u[(n / 2):,1], marker='o', color='r')
plt.title('f(u) = {0:.2}'.format(f(new_u)))
plt.show()

p = np.ones((n, n)) / (n - 1)
for i in xrange(n):
    p[i, i] = 0
dist_u = da.distributed_dual_averaging(u, fs, grads_f, prox, alphas, p)
for i in xrange(4):
    plt.subplot(2, 2, i)
    plt.scatter(x[:(n / 2),0], x[:(n / 2),1], marker='+', color='b')
    plt.scatter(x[(n / 2):,0], x[(n / 2):,1], marker='+', color='r')
    plt.scatter(dist_u[i][:(n / 2),0], dist_u[i][:(n / 2),1], marker='o', color='b')
    plt.scatter(dist_u[i][(n / 2):,0], dist_u[i][(n / 2):,1], marker='o', color='r')
    plt.title('f(u) = {0:.2}, i = {1}'.format(f(dist_u[i]), i))
plt.show()
