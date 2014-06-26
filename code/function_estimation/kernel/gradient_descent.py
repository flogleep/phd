import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def gen_1(n_sample):
    return sigma_1 * np.random.randn(n_sample, 2) + np.tile(mu_1, (n_sample, 1))

def gen_2(n_sample):
    return sigma_2 * np.random.randn(n_sample, 2) + np.tile(mu_2, (n_sample, 1))

def d(x, y):
    return np.sqrt(np.sum((x - y) * (x - y), axis=1))

def phi_p(x, y, a=-1., b=0):
    return (x[:, 1] - a * x[:, 0] - b) * (y[:, 1] - a * y[:, 0] - b) >= 0

def f_tilde(x, sample, d, phi_p):
    rep_x = np.tile(x, (sample.shape[0], 1))
    return np.mean(d(rep_x, sample) * phi_p(rep_x, sample))

def h(x, theta):
    return np.exp(-.5 * np.sum((x - theta) * (x - theta)))

def grad_h(x, theta):
    return (x - theta) * h(x, theta)

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

def r_tilde(thetas, ws, a, sample, ref):
    res = 0
    for i in xrange(sample.shape[0]):
        res += .5 * (big_h(sample[i, :], thetas, ws, a) - ref(sample[i, :])) ** 2
    return res

def grad_theta_r(thetas, ws, a, sample, ref):
    g = np.zeros(thetas.shape)
    ext_w = np.tile(ws, (thetas.shape[1], 1)).transpose()
    for i in xrange(sample.shape[0]):
        g += ext_w * grad_theta_big_h(sample[i, :], thetas, ws, a) \
            * (big_h(sample[i, :], thetas, ws, a) - ref(sample[i, :]))
    return g

def grad_w_r(thetas, ws, a, sample, ref):
    g = np.zeros(ws.shape)
    for i in xrange(sample.shape[0]):
        g += np.array([h(sample[i, :], thetas[l, :]) for l in xrange(thetas.shape[0])]) \
            * (big_h(sample[i, :], thetas, ws, a) - ref(sample[i, :]))
    return g

def grad_a_r(thetas, ws, a, sample, ref):
    g = 0
    for i in xrange(sample.shape[0]):
        g += big_h(sample[i, :], thetas, ws, a) - ref(sample[i, :])
    return g

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
sample = np.vstack((sample_1, sample_2))

def ref(x):
    return f_tilde(x, sample, d, phi_p)

print 'step 1'

n_it = 200
it = 1
temp_x = np.arange(min_x, max_x, 1.5)
temp_y = np.arange(min_y, max_y, 1.5)
list_thetas = []
for a in temp_x:
    for b in temp_y:
        list_thetas.append([a, b])
thetas = np.array(list_thetas)
#thetas = np.array([[1., 1.], [-1., -1.], [.5, .5], [-.5, -.5]])
print 'thetas : ' + str(thetas.shape)
ws = np.ones(thetas.shape[0]) / thetas.shape[0]
a = 0.
#steps = [.01 for i in xrange(1, 2 * n_it)]
eta = .9
values = [r_tilde(thetas, ws, a, sample, ref)]
print 'r_tilde = ' + str(values[0])
print 'step 2'
step = 10.
while it <= n_it:
    print str(it) + ' : ' + str(values[-1]) + ' (' + str(step) + ')'
    step /= eta
    g_theta = grad_theta_r(thetas, ws, a, sample, ref)
    g_w = grad_w_r(thetas, ws, a, sample, ref)
    g_a = grad_a_r(thetas, ws, a, sample, ref)
    new_thetas = thetas - step * g_theta
    new_ws = ws - step * g_w
    new_a = a - step * g_a
    v = r_tilde(new_thetas, new_ws, new_a, sample, ref)
    while v > values[-1]:
        step *= eta
        new_thetas = thetas - step * g_theta
        new_ws = ws - step * g_w
        new_a = a - step * g_a
        v = r_tilde(new_thetas, new_ws, new_a, sample, ref)
    thetas = new_thetas.copy()
    ws = new_ws.copy()
    a = new_a
    values.append(v)
    it += 1

#plt.plot(values)
#plt.show()

zs = np.array([big_h(np.array([x, y]), thetas, ws, a) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=True, linewidth=0, rstride=1, cstride=1)
ax.set_xlim(-2., 2.)
ax.set_ylim(-2., 2.)
ax.set_zlim(.2, .8)
ax = fig.add_subplot(223)
ax.imshow(Z)

zs = np.array([ref(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
u = Z.copy()
Z = zs.reshape(X.shape)

ax = fig.add_subplot(222, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=True, linewidth=0, rstride=1, cstride=1)
ax.set_xlim(-2., 2.)
ax.set_ylim(-2., 2.)
ax.set_zlim(.2, .8)
ax = fig.add_subplot(224)
levels = np.array([.1, .5, 1., 2.])
cs = ax.contour((u - Z) ** 2, levels)
ax.clabel(cs, inline=1, fontsize=10)
plt.show()
