"""Example of implementation of dual averaging."""

import numpy as np
import pdb
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


SIGMA_MC= .01
SIGMA_H = 1.
n_mc = 1000

def f_tilde(x, sample, d, phi_p):
    rep_x = np.tile(x, (sample.shape[0], 1))
    return np.mean(d(rep_x, sample) * phi_p(rep_x, sample))

def ref(x):
    return f_tilde(x, sample, d, phi_p)

def gen_1(n_sample):
    return sigma_1 * np.random.randn(n_sample, 2) + np.tile(mu_1, (n_sample, 1))

def gen_2(n_sample):
    return sigma_2 * np.random.randn(n_sample, 2) + np.tile(mu_2, (n_sample, 1))

def cond_expectation(est_f, mc_generator, x, n):
    means = np.zeros(x.shape[0])
    for i in xrange(x.shape[0]):
        means[i] = np.mean(est_f(np.tile(x[i, :], (n, 1)), mc_generator(n)))
    return means

def expectation(est_f, mc_generator, n):
    res = est_f(mc_generator(n))
    return np.mean(res, axis=(len(res.shape) - 1))

def d(x, y):
    if len(x.shape) > 1:
        return np.sum((x - y) ** 2, axis=1)
    else:
        return np.sum((x - y) ** 2)

def phi_p(x, y, a=-1., b=0):
    if len(x.shape) > 1:
        return (x[:, 1] - a * x[:, 0] - b) * (y[:, 1] - a * y[:, 0] - b) >= 0
    else:
        return (x[1] - a * x[0] - b) * (y[1] - a * y[0] - b) >= 0

def mc_generator(xs, n):
    k = xs.shape[0]
    mus = np.zeros((n, xs.shape[1]))
    for i in xrange(n):
        mus[i] = xs[np.random.randint(0, k)]
    return mus[i] + SIGMA_MC * np.random.randn(n, 2)

def h(x, theta):
    if len(x.shape) > 1:
        return np.exp(-.5 * np.sum((x - np.tile(theta, (x.shape[0], 1))) ** 2, axis=1)\
                / (SIGMA_H ** 2))
    else:
        return np.exp(-.5 * np.sum((x - theta) * (x - theta)) / (SIGMA_H ** 2))

def grad_h(x, theta):
    if len(x.shape) > 1:
        return -5 * (x - np.tile(theta, (x.shape[0], 1))) * np.tile(h(x, theta), (2, 1)).transpose()\
                / (SIGMA_H ** 2)
    else:
        return -.5 * (x - theta) * h(x, theta) / (SIGMA_H ** 2)

def big_h(x, thetas, ws, a):
    res = a
    for i in xrange(thetas.shape[0]):
        res += ws[i] * h(x, thetas[i, :])
    return res

def grad_theta_big_h(x, thetas, ws, a):
    g = np.zeros((thetas.shape[0], thetas.shape[1], x.shape[0]))
    for i in xrange(thetas.shape[0]):
        g[i, :, :] = grad_h(x, thetas[i, :]).transpose()
    return g

def grad_w_big_h(x, thetas, ws, a):
    return np.array([h(x, theta) for theta in thetas])

def grad_a_big_h(x, thetas, ws, a):
    return 1.

def grad_big_h(x, thetas, ws, a):
    return ((grad_theta_big_h(x, thetas, ws, a),
             grad_w_big_h(x, thetas, ws, a),
             grad_a_big_h(x, thetas, ws, a)))

def f(x, y):
    return d(x, y) * phi_p(x, y)

def r(thetas, ws, a, mc_generator):
    est_f = lambda x: (big_h(x, thetas, ws, a) - cond_expectation(f, mc_generator, x, n_mc)) ** 2
    return expectation(est_f, mc_generator, n_mc)

def grad_theta_r(thetas, ws, a, mc_generator):
    est_f = lambda x: grad_theta_big_h(x, thetas, ws, a) \
            * np.tile(big_h(x, thetas, ws, a), (thetas.shape[0], thetas.shape[1], 1))\
            - np.tile(cond_expectation(f, mc_generator, x, n_mc), (thetas.shape[0], thetas.shape[1], 1))
    return expectation(est_f, mc_generator, n_mc)

def grad_w_r(thetas, ws, a, mc_generator):
    est_f = lambda x: grad_w_big_h(x, thetas, ws, a) \
            * big_h(x, thetas, ws, a) - cond_expectation(f, mc_generator, x, n_mc)
    return expectation(est_f, mc_generator, n_mc)

def grad_a_r(thetas, ws, a, mc_generator):
    est_f = lambda x: grad_a_big_h(x, thetas, ws, a) \
            * big_h(x, thetas, ws, a) - cond_expectation(f, mc_generator, x, n_mc)
    return expectation(est_f, mc_generator, n_mc)

def grad_r(thetas, ws, a, mc_generator):
    return ((grad_theta_r(x, thetas, ws, a),
             grad_w_r(x, thetas, ws, a),
             grad_a_r(x, thetas, ws, a)))

mu_1 = np.array([1., 1.])
mu_2 = np.array([-1., -1.])
sigma_1 = .5
sigma_2 = .5

max_x = 3.
max_y = 3.
min_x = -3.
min_y = -3.
dx = .1
dy = .1

x = np.arange(min_x, max_x, dx)
y = np.arange(min_y, max_y, dy)
X, Y = np.meshgrid(x, y)

n = 20
sample_1 = gen_1(n / 2)
sample_2 = gen_2(n / 2)
#xs_0 = np.vstack((sample_1[:(n/4),:], sample_2[:(n/4),:]))
xs_0 = np.vstack((sample_1, sample_2))
xs_1 = np.vstack((sample_1[(n/4):,:], sample_2[(n/4):,:]))
mc_gen_0 = lambda n: mc_generator(xs_0, n)
mc_gen_1 = lambda n: mc_generator(xs_1, n)


sample = np.vstack((sample_1, sample_2))

n_iter = 100
k = .01
steps = [k / np.sqrt(t + 1) for t in xrange(n_iter + 1)]
p = np.array([[1., 0.], [0., 1.]])

theta_0 = np.array([[.5, 1.], [1., 3.]])
av_t_0 = theta_0.copy()
w_0 = .5 * np.ones(2)
av_w_0 = w_0.copy()
a_0 = 1.
av_a_0 = a_0
theta_1 = -theta_0.copy()
av_t_1 = theta_1.copy()
w_1 = .5 * np.ones(2)
av_w_1 = w_1.copy()
a_1 = 1.
av_a_1 = a_1

for i in xrange(10):
    print r(theta_0, w_0, a_0, mc_gen_0)
pdb.set_trace()

it = 0
rs_0 = []
rs_1 = []
while it <= n_iter:
    grad_theta_0 = grad_theta_r(theta_0, w_0, a_0, mc_gen_0)
    grad_theta_1 = grad_theta_r(theta_1, w_1, a_1, mc_gen_1)
    grad_w_0 = grad_w_r(theta_0, w_0, a_0, mc_gen_0)
    grad_w_1 = grad_w_r(theta_1, w_1, a_1, mc_gen_1)
    grad_a_0 = grad_a_r(theta_0, w_0, a_0, mc_gen_0)
    grad_a_1 = grad_a_r(theta_1, w_1, a_1, mc_gen_1)

    theta_0 -= steps[it] * (p[0, 0] * grad_theta_0 + p[0, 1] * grad_theta_1)
    theta_1 -= steps[it] * (p[1, 0] * grad_theta_0 + p[1, 1] * grad_theta_1)
    w_0 -= steps[it] * (p[0, 0] * grad_w_0 + p[0, 1] * grad_w_1)
    w_1 -= steps[it] * (p[1, 0] * grad_w_0 + p[1, 1] * grad_w_1)
    a_0 -= steps[it] * (p[0, 0] * grad_a_0 + p[0, 1] * grad_a_1)
    a_1 -= steps[it] * (p[1, 0] * grad_a_0 + p[1, 1] * grad_a_1)

    rs_0.append(r(theta_0, w_0, a_0, mc_gen_0))
    rs_1.append(r(theta_1, w_1, a_1, mc_gen_1))

    print str(it) + ": " + str(rs_0[-1]) + ", " + str(rs_1[-1])
    it += 1

    av_t_0 = ((it + 1.) / (it + 2.)) * av_t_0 + (1. / (it + 2.)) * theta_0
    av_t_1 = ((it + 1.) / (it + 2.)) * av_t_1 + (1. / (it + 2.)) * theta_1
    av_w_0 = ((it + 1.) / (it + 2.)) * av_w_0 + (1. / (it + 2.)) * w_0
    av_w_1 = ((it + 1.) / (it + 2.)) * av_w_1 + (1. / (it + 2.)) * w_1
    av_a_0 = ((it + 1.) / (it + 2.)) * av_a_0 + (1. / (it + 2.)) * a_0
    av_a_1 = ((it + 1.) / (it + 2.)) * av_a_1 + (1. / (it + 2.)) * a_1

plt.subplot(121)
plt.plot(rs_0)
plt.subplot(122)
plt.plot(rs_1)
plt.show()

zs = np.array([big_h(np.array([x, y]), av_t_0, av_w_0, av_a_0) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=0., vmax=1.5,
                antialiased=True, linewidth=0, rstride=1, cstride=1)
ax.set_xlim(-2., 2.)
ax.set_ylim(-2., 2.)
ax.set_zlim(.2, .8)
ax = fig.add_subplot(223)
ax.imshow(Z)

zs = np.array([ref(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
u = Z.copy()
Z = zs.reshape(X.shape)

ax = fig.add_subplot(222, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=0., vmax=1.5,
                antialiased=True, linewidth=0, rstride=1, cstride=1)
ax.set_xlim(-2., 2.)
ax.set_ylim(-2., 2.)
ax.set_zlim(.2, .8)
ax = fig.add_subplot(224)
levels = np.array([.005, .01, .1, .5, 1., 2.])
cs = ax.contour((u - Z) ** 2, levels)
ax.clabel(cs, inline=1, fontsize=10)
plt.show()
