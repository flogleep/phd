import dual_averaging as da
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math

n = 2 * 10

x = np.vstack([.5 * np.random.randn(n / 2, 2) + np.array([3., 3.]),
               .5 * np.random.randn(n / 2, 2) + np.array([-3., -3.])
               ])

spr = 10.0 / (n * n)

def mega_norm(u):
    """Return norm1 of differences."""

    n = u.shape[0]
    norms = np.zeros((n, n))
    for i in xrange(n):
        rep_ui = np.tile(u[i], (n, 1))
        abs_u = np.abs(u - rep_ui)
        norms[i] = np.sum(abs_u, axis=1)

    return norms


def dist_relaxed_clustering(i, u, x, spr, w):
    """Return value of f_i."""

    return .5 * np.sum((u[i] - x[i]) ** 2) + spr * np.sum(w[i] * mega_norm(u)[i])

def dist_grad_relaxed(i, u, x, spr, w):
    """Return grad of f_i."""

    n = u.shape[0]
    g = np.zeros(u.shape)
    s = ((np.tile(u[i], (n, 1)) < u)).astype('int')\
            - ((np.tile(u[i], (n, 1)) > u)).astype('int')
    g += spr * np.diag(w[i]).dot(s)
    g[i] += spr * w[i].dot(-s) + (u[i] - x[i])

    return g

def relaxed_clustering(u, x, spr, w):
    """Return value of f."""

    return .5 * np.sum((u - x) ** 2) + spr * np.sum(w * mega_norm(u))

def grad_relaxed(u, x, spr, w):
    """Return gradient of f."""

    n = u.shape[0]
    g = np.zeros(u.shape)
    for i in xrange(n):
        s = ((np.tile(u[i], (n, 1)) < u)).astype('int')\
                - ((np.tile(u[i], (n, 1)) > u)).astype('int')
        g += spr * np.diag(w[i]).dot(s)
        g[i] += spr * w[i].dot(-s) + (u[i] - x[i])

    return g


d = mega_norm(x)
eps = 1e-8
w1 = 1 / np.maximum(d, eps * np.ones(d.shape))
w1 = w1 / np.linalg.norm(w1)
w2 = np.zeros((n, n))
w2[:(n / 2), :(n / 2)] = 1
w2[(n / 2):, (n / 2):] = 1
w2 = w2 / np.linalg.norm(w2)
w = w1
f = lambda u: relaxed_clustering(u, x, spr, w)
fs = lambda i, u: dist_relaxed_clustering(i, u, x, spr, w)
grad_f = lambda u: grad_relaxed(u, x, spr, w)
grads_f = lambda i, u: dist_grad_relaxed(i, u, x, spr, w)
u = x.copy()
prox = lambda z, a: -a * z
alphas = [1. / math.sqrt(t) for t in xrange(1, 1000)]

#min_bound = -6.
#max_bound = 6.
#step = .1
#z_x = np.arange(min_bound, max_bound, step)
#z_y = z_x.copy()
#xx, yy = np.meshgrid(z_x, z_y)
#z = np.zeros(xx.shape)
#test_u = u.copy()
#for i in xrange(z.shape[0]):
#    for j in xrange(z.shape[1]):
#        test_u[0] = np.array([xx[i, j], yy[i, j]])
#        z[i, j] = f(test_u)
#
#print "x0 : ", x[0]
#print "||x - u||/2 : ", .5 * np.sum((x - u) ** 2)
#print "f(u) : ", f(u)
#
#plt.subplot(121)
#p = plt.imshow(z)
#cbar = plt.colorbar(p)
#cbar.ax.set_ylabel('f value')
#plt.subplot(122)
#plt.scatter(x[:(n / 2),0], x[:(n / 2),1], color='b')
#plt.scatter(x[(n / 2):,0], x[(n / 2):,1], color='r')
#plt.xlim([min_bound, max_bound])
#plt.ylim([min_bound, max_bound])
#plt.show()

#new_u, values1 = da.dual_averaging(u, f, grad_f, prox, alphas, 20)
#plt.scatter(x[:(n / 2),0], x[:(n / 2),1], marker='+', color='b')
#plt.scatter(x[(n / 2):,0], x[(n / 2):,1], marker='+', color='r')
#plt.scatter(new_u[:(n / 2),0], new_u[:(n / 2),1], marker='o', color='b')
#plt.scatter(new_u[(n / 2):,0], new_u[(n / 2):,1], marker='o', color='r')
#plt.title('f(u) = {0:.2}'.format(f(new_u)))
#plt.show()
#
#w = w2
#new_u, values2 = da.dual_averaging(u, f, grad_f, prox, alphas, 20)
#plt.scatter(x[:(n / 2),0], x[:(n / 2),1], marker='+', color='b')
#plt.scatter(x[(n / 2):,0], x[(n / 2):,1], marker='+', color='r')
#plt.scatter(new_u[:(n / 2),0], new_u[:(n / 2),1], marker='o', color='b')
#plt.scatter(new_u[(n / 2):,0], new_u[(n / 2):,1], marker='o', color='r')
#plt.title('f(u) = {0:.2}'.format(f(new_u)))
#plt.show()
#
#plt.plot(values1, linewidth=3, label='w = 1/d')
#plt.plot(values2, linewidth=3, label='w = Phi_p')
#plt.legend()
#plt.xlabel('Iteration')
#plt.ylabel('f')
#plt.show()



#p = np.ones((n, n)) / (n - 1)
p = np.zeros((n, n))
p[(n / 4):(3 * n / 4), (n / 4):(3 * n / 4)] = 1. / (n / 2)

for i in xrange(n):
    p[i, i] = 0
w = w1
dist_u, values1 = da.distributed_dual_averaging(u, fs, grads_f, prox, alphas, p)
w = w2
dist_u, values2 = da.distributed_dual_averaging(u, fs, grads_f, prox, alphas, p)
plt.plot(values1, linewidth=3, label='w = 1/d')
plt.plot(values2, linewidth=3, label='w = Phi_p')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('f')
plt.show()
for i in xrange(4):
    plt.subplot(2, 2, i)
    plt.scatter(x[:(n / 2),0], x[:(n / 2),1], marker='+', color='b')
    plt.scatter(x[(n / 2):,0], x[(n / 2):,1], marker='+', color='r')
    plt.scatter(dist_u[i][:(n / 2),0], dist_u[i][:(n / 2),1], marker='o', color='b')
    plt.scatter(dist_u[i][(n / 2):,0], dist_u[i][(n / 2):,1], marker='o', color='r')
    plt.title('f(u) = {0:.2}, i = {1}'.format(f(dist_u[i]), i))
plt.show()
