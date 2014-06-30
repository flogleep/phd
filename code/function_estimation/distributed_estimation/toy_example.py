"""Example of implementation of dual averaging."""

import numpy as np
import pdb
import matplotlib.pyplot as plt


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

def f(x, y):
    return d(x, y) * phi_p(x, y)

def r(thetas, ws, a, mc_generator):
    est_f = lambda x: (big_h(x, thetas, ws, a) - cond_expectation(f, mc_generator, x, n_mc)) ** 2
    return expectation(est_f, mc_generator, n_mc)

def grad_theta_r(thetas, ws, a, mc_generator):
    est_f = lambda x: grad_theta_big_h(x, thetas, ws, a) \
            * big_h(x, thetas, ws, a) - cond_expectation(f, mc_generator, x, n_mc)

