"""Module for dual averaging implementation."""

import numpy as np


class Agent(object):
    """Agent used for distributed optimization."""

    def __init__(self, f, grad_f, prox, x0, param=None):
        self.param = param
        if param is None:
            self.f = f
            self.grad_f = grad_f
        else:
            self.f = lambda x: f(x, *param)
            self.grad_f = lambda x: grad_f(x, *param)
        self.prox = prox
        self.xs = np.reshape(x0, (1, x0.shape[0]))

        self.update_z()

    def update_z(self, other_zs):
        new_z = other_zs + self.grad_f(self.xs[-1])
        self.zs = np.vstack((self.zs, new_z.copy()))

        return new_z

    def update_x(self, step):
        new_x = self.prox(self.zs[-1], step)
        self.xs = np.vstack((self.xs, new_x.copy()))

        return new_x

    def last_z(self):
        return self.zs[-1].copy()


class Network(object):
    """Controller for distributed optimization."""

    def __init__(self, fs, grad_fs, prox, x0, steps, p):
        self.agents = []
        self.nb_agents = len(fs)
        self.last_zs = np.zeros((self.nb_agents, x0.shape[0]))
        for i in xrange(self.nb_agents):
            self.agents.append(Agent(fs[i], grad_fs[i], prox, x0))
            self.last_zs[i] = self.agents[i].last_z()
        self.steps = steps
        self.p = p
        self.current_iteration = 0

    def next_iteration(self):
        updated_zs = np.zeros(self.last_zs.shape)
        other_zs = self.p.dot(self.last_zs)
        for i in xrange(self.nb_agents):
            updated_zs[i] = self.agents[i].update_z(other_zs[i])
            self.agents[i].update_x(self.steps[self.current_iteration])
        self.last_zs = updated_zs.copy()
        self.current_iteration += 1
