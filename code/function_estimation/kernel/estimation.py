"""Methods and toy example for kernel-based function estimation."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


mu_1 = np.array([1., 1.])
mu_2 = np.array([-1., -1.])
sigma_1 = .5
sigma_2 = .5

max_x = 3.
max_y = 3.
min_x = -3.
min_y = -3.
dx = .05
dy = .05

x = np.arange(min_x, max_x, dx)
y = np.arange(min_y, max_y, dy)
X, Y = np.meshgrid(x, y)

def gen_1(n_sample):
    return sigma_1 * np.random.randn(n_sample, 2) + np.tile(mu_1, (n_sample, 1))

def gen_2(n_sample):
    return sigma_2 * np.random.randn(n_sample, 2) + np.tile(mu_2, (n_sample, 1))

def d(x, y):
    return np.sqrt(np.sum((x - y) * (x - y), axis=1))

def phi_p(x, y, a, b):
    return (x[:, 1] - a * x[:, 0] - b) * (y[:, 1] - a * y[:, 0] - b) >= 0

def mc_estimation(x, y, a, b, n):
    w = np.tile(np.array([x, y]), (n / 2, 1))
    sample_1 = gen_1(n / 2)
    sample_2 = gen_2(n / 2)
    s = 0
    s += np.sum(d(w, sample_1) * phi_p(w, sample_1, a, b))
    s += np.sum(d(w, sample_2) * phi_p(w, sample_2, a, b))
    return s / n

def plot_image(X, Y, a=-1, b=0, n=1000, ax=None):
    zs = np.array([mc_estimation(x, y, a, b, n) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    if ax is None:
        fig = plt.figure()
        #2D
        ax = fig.add_subplot(111)
        #3D
        #ax = fig.add_subplot(111, projection='3d')

    #3D visualization
    #ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=True, linewidth=0, rstride=1, cstride=1)

    #2D visualization
    ax.imshow(Z)
    ax.plot()


plot_image(X, Y)
plt.show()
plot_image(X, Y, a=-1.5, b=.5, n=1000)
plt.show()
