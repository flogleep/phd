"""Quick example of k-means application."""


import numpy as np
import matplotlib.pyplot as plt
import pdb
from random import choice


def assignation(points, centroids):
    """Return a vector of closest centroids indexes."""

    clusters = np.zeros(points.shape[0])
    for i in xrange(points.shape[0]):
        index = -1
        min_dist = np.inf
        for k in xrange(centroids.shape[0]):
            dist = np.sum((points[i] - centroids[k]) ** 2)
            if dist < min_dist:
                index = k
                min_dist = dist
        clusters[i] = index

    return clusters

def centroids_update(points, clusters, nb_centroids):
    """Return new centroids according to clusters assignations."""

    centroids = np.zeros((nb_centroids, points.shape[1]))
    for k in xrange(nb_centroids):
        count = 0
        for i in xrange(points.shape[0]):
            if clusters[i] == k:
                count += 1
                centroids[k] += points[i]
        centroids[k] /= count

    return centroids


mu1 = [1., 1.]
mu2 = [-1., -1.]
sigma1 = .7
sigma2 = .7
n = 2 * 60
d = 2
nb_clust = 2

b_x = (.5, .5, 1.)
r_x = (1., .5, .5)
b_f = (.98, .98, 1.)
r_f = (1., .98, .98)

x1 = sigma1 * np.random.randn(n / 2, d) + mu1
x2 = sigma2 * np.random.randn(n / 2, d) + mu2
points = np.vstack([x1, x2])

c = np.zeros((nb_clust, d))
for i in xrange(nb_clust):
    ind = choice(range(n))
    c[i] = points[ind]

plt.scatter(x1[:, 0], x1[:, 1], s=20, color=b_x, marker='+')
plt.scatter(x2[:, 0], x2[:, 1], s=20, color=r_x, marker='+')
plt.xlim(-3., 3.)
plt.ylim(-3., 3.)
plt.savefig('./kmeans_regular_0.pdf')

clusters = assignation(points, c)
clust1 = np.zeros((1, d))
clust2 = np.zeros((1, d))
for i in xrange(n):
    if clusters[i] == 0:
        clust1 = np.vstack([clust1, points[i]])
    else:
        clust2 = np.vstack([clust2, points[i]])

plt.scatter(clust1[1:, 0], clust1[1:, 1], s=20, color=b_x, marker='+')
plt.scatter(clust2[1:, 0], clust2[1:, 1], s=20, color=r_x, marker='+')
plt.scatter(c[0, 0], c[0, 1], s=350, alpha=.5, color='b')
plt.scatter(c[1, 0], c[1, 1], s=350, alpha=.5, color='r')
plt.xlim(-3., 3.)
plt.ylim(-3., 3.)
plt.savefig('./kmeans_regular_1.pdf')


niter_max = 4
for it in xrange(niter_max):
    clusters = assignation(points, c)

    clust1 = np.zeros((1, d))
    clust2 = np.zeros((1, d))
    for i in xrange(n):
        if clusters[i] == 0:
            clust1 = np.vstack([clust1, points[i]])
        else:
            clust2 = np.vstack([clust2, points[i]])

    direc = np.array([c[1, 1] - c[0, 1], c[0, 0] - c[1, 0]])
    p1 = .5 * (c[1, :] + c[0, :]) + 10000 * direc
    p2 = .5 * (c[1, :] + c[0, :]) - 10000 * direc
    d_c0 = np.sum((np.array([-3., -10.]) - c[0, :]) ** 2)
    d_c1 = np.sum((np.array([-3., -10.]) - c[1, :]) ** 2)
    if d_c0 < d_c1:
        col_bot = b_f
        col_top = r_f
    else:
        col_bot = r_f
        col_top = b_f

    plt.clf()
    plt.fill_between([p1[0], p2[0]], [p1[1], p2[1]], -10, color=col_bot)
    plt.fill_between([p1[0], p2[0]], [p1[1], p2[1]], 10, color=col_top)
    plt.scatter(clust1[1:, 0], clust1[1:, 1], s=20, color=b_x, marker='+')
    plt.scatter(clust2[1:, 0], clust2[1:, 1], s=20, color=r_x, marker='+')
    plt.scatter(c[0, 0], c[0, 1], s=350, alpha=.5, color='b')
    plt.scatter(c[1, 0], c[1, 1], s=350, alpha=.5, color='r')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linewidth=2)
    plt.xlim(-3., 3.)
    plt.ylim(-3., 3.)
    plt.savefig('./kmeans_regular_{0}.pdf'.format(3 * (it + 1) - 1))


    plt.clf()
    prev_c = c.copy()
    c = centroids_update(points, clusters, nb_clust)
    plt.fill_between([p1[0], p2[0]], [p1[1], p2[1]], -10, color=col_bot)
    plt.fill_between([p1[0], p2[0]], [p1[1], p2[1]], 10, color=col_top)
    plt.scatter(clust1[1:, 0], clust1[1:, 1], s=20, color=b_x, marker='+')
    plt.scatter(clust2[1:, 0], clust2[1:, 1], s=20, color=r_x, marker='+')
    plt.scatter(prev_c[0, 0], prev_c[0, 1], s=350, alpha=.2, color='b')
    plt.scatter(prev_c[1, 0], prev_c[1, 1], s=350, alpha=.2, color='r')
    plt.scatter(c[0, 0], c[0, 1], s=350, alpha=.5, color='b')
    plt.scatter(c[1, 0], c[1, 1], s=350, alpha=.5, color='r')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linewidth=2)
    plt.xlim(-3., 3.)
    plt.ylim(-3., 3.)
    plt.savefig('./kmeans_regular_{0}.pdf'.format(3 * (it + 1)))

    plt.clf()
    plt.fill_between([p1[0], p2[0]], [p1[1], p2[1]], -10, color=col_bot)
    plt.fill_between([p1[0], p2[0]], [p1[1], p2[1]], 10, color=col_top)
    plt.scatter(clust1[1:, 0], clust1[1:, 1], s=20, color=b_x, marker='+')
    plt.scatter(clust2[1:, 0], clust2[1:, 1], s=20, color=r_x, marker='+')
    plt.scatter(c[0, 0], c[0, 1], s=350, alpha=.5, color='b')
    plt.scatter(c[1, 0], c[1, 1], s=350, alpha=.5, color='r')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linewidth=2)
    plt.xlim(-3., 3.)
    plt.ylim(-3., 3.)
    plt.savefig('./kmeans_regular_{0}.pdf'.format(3 * (it + 1) + 1))
