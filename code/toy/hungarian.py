"""Toy example for hungarian algorithm implementation."""

import numpy as np
#import scipy as sp
#import math


def zero_rows(m):
    """Return the matrix where minimal elements of each row are substracted.

    There is at least one 0 by row.

    """

    return m - np.outer(np.min(m, axis=1), np.ones(m.shape[1]))


def zero_columns(m):
    """Return the matrix where minimal elements of each column are substracted.

    There is at least one 0 by column.

    """
    return m - np.outer(np.ones(m.shape[0]), np.min(m, axis=0))


a = np.array(
    [
        [1, 2, 3],
        [2, 4, 1],
        [3, 6, 2]
    ])

print a
print zero_rows(a)
print zero_columns(a)
