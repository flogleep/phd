import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def k(x):
    if len(x.shape) > 1:
        return np.exp(-.5 * np.sum(x ** 2, axis=1))
    return np.exp(-.5 * (x ** 2))

x_min = -5.
x_max = 5.
y_min = -.5
y_max = 1.5
dx = .1
x = np.arange(x_min, x_max, dx)
ax = plt.subplot(111)
ax.plot(x, k(x), linewidth=4)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.axis('off')
plt.show()

X, Y = np.meshgrid(x, x)
zs = k(np.array(zip(X.ravel(), Y.ravel())))
Z = zs.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=0., vmax=1.5,
                antialiased=True, linewidth=0, rstride=1, cstride=1)
ax.set_xlim(-5., 5.)
ax.set_ylim(-5., 5.)
ax.set_zlim(-.5, 1.5)
ax.axis('off')
plt.show()
