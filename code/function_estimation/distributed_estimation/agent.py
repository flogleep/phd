"""Module for dual averaging implementation."""

import numpy as np


class Agent(object):
    """Agent used for distributed optimization."""

    def __init__(self, f_mat, grad_f_mat,
                 f_vec, grad_f_vec,
                 f_scal, grad_f_scal,
                 prox,
                 mat_0, vec_0, scal_0,
                 param=None):
        self.param = param
        if param is None:
            self.f_mat = f_mat
            self.f_vec = f_vec
            self.f_scal = f_scal
            self.grad_f_mat = grad_f_mat
            self.grad_f_vec = grad_f_vec
            self.grad_f_scal = grad_f_scal
        else:
            self.f_mat = lambda x: f_mat(x, *param)
            self.f_vec = lambda x: f_vec(x, *param)
            self.f_scal = lambda x: f_scal(x, *param)
            self.grad_f_mat = lambda x: grad_f_mat(x, *param)
            self.grad_f_vec = lambda x: grad_f_vec(x, *param)
            self.grad_f_scal = lambda x: grad_f_scal(x, *param)
        self.prox = prox
        self.xs_mat = [mat_0]
        self.xs_vec = [vec_0]
        self.xs_scal = [scal_0]

        self.zs_mat = [np.zeros(mat_0.shape)]
        self.zs_vec = [np.zeros(vec_0.shape)]
        self.zs_scal = [0]

    def update_z(self, other_zs_mat, other_zs_vec, other_zs_scal):
        new_z_mat = other_zs_mat + self.grad_f_mat(self.xs_mat[-1])
        self.zs_mat.append(new_z_mat.copy())
        new_z_vec = other_zs_vec + self.grad_f_vec(self.xs_vec[-1])
        self.zs_vec.append(new_z_vec.copy())
        new_z_scal = other_zs_scal + self.grad_f_scal(self.xs_scal[-1])
        self.zs_scal.append(new_z_scal.copy())

        return (new_z_mat, new_z_vec, new_z_scal)

    def update_x(self, step):
        new_x_mat = self.prox(self.zs_mat[-1], step)
        self.xs_mat.append(new_x_mat.copy())
        new_x_vec = self.prox(self.zs_vec[-1], step)
        self.xs_vec.append(new_x_vec.copy())
        new_x_scal = self.prox(self.zs_scal[-1], step)
        self.xs_scal.append(new_x_scal.copy())

        return (new_x_mat, new_x_vec, new_x_scal)

    def last_z(self):
        return (self.zs_mat[-1].copy(),
                self.zs_vec[-1].copy(),
                self.zs_scal[-1].copy())


class Network(object):
    """Controller for distributed optimization."""

    def __init__(self,
                 fs_mat, fs_vec, fs_scal,
                 grad_fs_mat, grad_fs_vec, grad_fs_scal,
                 prox,
                 mat_0, vec_0, scal_0,
                 steps, p):
        self.agents = []
        self.nb_agents = len(fs_mat)
        self.last_zs_mat = np.zeros((self.nb_agents, mat_0.shape[0]))
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
