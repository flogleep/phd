import numpy as np
import matplotlib.pyplot as plt
import pdb

def f(x):
  return 3 - 100*x - 2*(x + 2)**2 + .1 * (x - 2)**3

x_min = -10.
x_max = 10.
y_min = -1500.
y_max = 1000.
dx = .01
x = np.arange(x_min, x_max, dx)
ax = plt.subplot(111)
ax.plot(x, f(x), linewidth=4)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.savefig('./figures/regression_f.pdf')
plt.clf()
ax = plt.subplot(111)
ax.plot(x, f(x), linewidth=4, linestyle='--', alpha=.5)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
obs = x_min + (x_max - x_min) * np.random.random(40)
ax.scatter(obs, f(obs), color='r', marker='.', s=200)
plt.savefig('./figures/regression_obs.pdf')
#ax.axis('off')

def grad_r(theta, f_hat, sample_x, sample_y):
  g = np.zeros(theta.shape)
  for i in xrange(sample_x.shape[0]):
    g += (f_hat(sample_x[i], theta) - sample_y[i])\
        * np.array([1., sample_x[i], sample_x[i] ** 2])
  return g / sample_x.shape[0]

def f_hat(x, theta):
  s = 0
  for i in xrange(theta.shape[0]):
    s += theta[i] * (x ** i)
  return s

def r(theta, f_hat, sample_x, sample_y):
  s = 0
  for i in xrange(sample_x.shape[0]):
    s += (f_hat(sample_x[i], theta) - sample_y[i]) ** 2 
  return s / (2 * sample_x.shape[0])

n_iter = 100
step = [.0005 for i in xrange(n_iter)]

theta = np.array([0., 0., 0.])
theta0 = theta.copy()
rs = [r(theta, f_hat, obs, f(obs))]
for i in xrange(n_iter):
  if i == 10:
    theta1 = theta.copy()
  if i == 50:
    theta2 = theta.copy()
  theta -= step[i] * grad_r(theta, f_hat, obs, f(obs))
  rs.append(r(theta, f_hat, obs, f(obs)))

plt.clf()
ax = plt.subplot(111)
ax.plot(x, f(x), linewidth=4, linestyle='--', alpha=.5)
ax.scatter(obs, f(obs), color='r', marker='.', s=200)
ax.plot(x, f_hat(x, theta0), linewidth=4, color='g', alpha=.5)
ax.plot(x, f_hat(x, theta1), linewidth=4, color='g', alpha=.5)
ax.plot(x, f_hat(x, theta2), linewidth=4, color='g', alpha=.5)
ax.plot(x, f_hat(x, theta), linewidth=4, color='g', alpha=.5)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.savefig('./figures/regression_several-thetas.pdf')
plt.show()
