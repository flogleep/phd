from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

dx = .005
dy = .005
y, x = np.mgrid[slice(0, 1, dy), slice(0, 1, dx)]
z = np.absolute(x - y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z)
plt.show()
