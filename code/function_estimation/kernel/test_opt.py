import numpy as np
import pdb


def mega_norm(u):
    """Return norm1 of differences."""

    n = u.shape[0]
    norms = np.zeros((n, n))
    for i in xrange(n):
        rep_ui = np.tile(u[i], (n, 1))
        abs_u = np.abs(u - rep_ui)
        norms[i] = np.sum(abs_u, axis=1)
    print norms

u = np.array([[1., 0.],
              [1., 1.],
              [2., 3.]])

S = (np.tile(u[0], (3, 1)) < u).astype('int')
w = np.array([1, 2, 3])
print S
print np.diag(w)
print np.diag(w).dot(S)
