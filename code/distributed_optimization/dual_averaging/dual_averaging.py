"""Implementation of dual averaging algorithm."""

import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

def dual_averaging(x0, f, grad_f, prox, alphas):
    """Return result of dual averaging optimization algorithm.

    Inputs:
        x0     : starting location
        f      : objective function
        grad_f : gradient of objective function (or in case of
                 non smooth optimization, function that returns a subgradient
                 of the objective function)
        prox   : proximal projection operator
        alphas : steps sequence

    Outputs:
        TBD

    """

    x = x0
    z = x.copy()
    #TODO: implement a smart stopping criterion
    n_iter_max = 100
    current_iter = 0
    stop = False
    values = [f(x0)]

    u0s = np.zeros((n_iter_max + 1, 2))
    u0s[0] = x[0].copy()

    while not stop:
        current_iter += 1
        alpha = alphas[min(len(alphas), current_iter) - 1]

        g = grad_f(x)
        z += g
        x = prox(z, alpha)
        values.append(f(x))
        u0s[current_iter] = x[0].copy()

        stop = current_iter >= n_iter_max

    plt.subplot(121)
    plt.plot(u0s[:, 0], u0s[:, 1], 'o-', linewidth=3, color='r')
    plt.subplot(122)
    plt.plot(values)
    plt.show()
    return x


def distributed_dual_averaging(x0, fs, grads_f, prox, alphas, p):
    """Return result of distributed dual averaging algorithm.

    Inputs:
        x0      : starting location
        fs      : list of objective functions
        grads_f : list of respective gradients of objective functions
                  (or in case of non smooth optimization, functions that
                  return a subgradient of the objective functions)
        prox    : proximal projection operator
        alphas  : steps sequence
        p       : weight matrix (must be doubly stochastic)

    Outputs:
        TBD

    """

    n = p.shape[0]
    xs = [x0.copy() for i in xrange(n)]
    av_xs = [x0.copy() for i in xrange(n)]
    zs = [x0.copy() for i in xrange(n)]
    #TODO: implement a smart stopping criterion
    n_iter_max = 100
    current_iter = 0
    stop = False

    while not stop:
        current_iter += 1
        alpha = alphas[min(len(alphas), current_iter) - 1]

        for i in xrange(n):
            gi = grads_f(i, xs[i])
            zs[i] = gi.copy()
            for k in xrange(len(p[i])):
                zs[i] += p[i, k] * zs[k]
            xs[i] = prox(zs[i], alpha)
            av_xs[i] = (current_iter * av_xs[i] + xs[i]) / (current_iter + 1)

        stop = current_iter >= n_iter_max

    return av_xs
