"""Example of implementation of dual averaging."""

import numpy as np
import pdb
import matplotlib.pyplot as plt
from agent import *


SIGMA = 1.
n_mc = 500

def cond_expectation(est_f, mc_generator, x, n):
    s = 0
    for i in xrange(n):
        s += est_f(x, mc_generator())
    return s / n

def expectation(est_f, mc_generator, n):
    s = 0
    for i in xrange(n):
        s += est_f(mc_generator())
    return s / n

def d(x, y):
    return np.sqrt(np.sum((x - y) * (x - y), axis=1))

def phi_p(x, y, a=-1., b=0):
    return (x[:, 1] - a * x[:, 0] - b) * (y[:, 1] - a * y[:, 0] - b) >= 0

def mc_generator(xs):
    k = xs.shape[0]
    i = np.random.randint(0, k)
    return xs[i] + SIGMA * np.random.randn(2)

def h(x, theta):
    return np.exp(-.5 * np.sum((x - theta) * (x - theta)) / (SIGMA ** 2))

def grad_h(x, theta):
    return -.5 * (x - theta) * h(x, theta) / (SIGMA ** 2)

def big_h(x, thetas, ws, a):
    res = a
    for i in xrange(thetas.shape[0]):
        res += ws[i] * h(x, thetas[i, :])
    return res

def grad_theta_big_h(x, thetas, ws, a):
    g = np.zeros(thetas.shape)
    for i in xrange(thetas.shape[0]):
        g[i, :] = grad_h(x, thetas[i, :])
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
            * big_h(x, thetas, ws, a) - cond_expectation(f, mc_generator, x, n_mc)
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

n = 200
sample_1 = gen_1(n / 2)
sample_2 = gen_2(n / 2)
xs_0 = np.vstack((sample_1[:(n/4),:], sample_2[:(n/4),:]))
xs_1 = np.vstack((sample_1[(n/4):,:], sample_2[(n/4):,:]))
mc_gen_0 = lambda : mc_generator(xs_0)
mc_gen_1 = lambda : mc_generator(xs_1)
prox : lambda z, alpha: -alpha * z
agent_0 = Agent(r, grad_r,


sample = np.vstack((sample_1, sample_2))
