"""Backtracking functions for one optimization algorithm step."""

import numpy as np
import pdb


def is_legal(v):
    """Return whether or not a step is valid."""
    return np.sum(np.imag(v)) == 0 and not np.isnan(v.flatten()).any()\
        and not np.isinf(v.flatten()).any()

def backtracking(x, t, d, f, g, fr, gtd, c1,
                 debug=False,
                 ls_interp=0,
                 ls_multi=0,
                 prog_tol=1e-3,
                 do_plot=False,
                 fun_obj=None,
                 *varargin):
    """
    Return backtracked linesearch with respect to Armijo condition.

    Inputs:
        x:          starting location
        t:          initial stepsize
        d:          descent direction
        f:          function value at starting location
        fr:         reference function value
        gtd:        directional derivative at starting location
        c1:         sufficient decrease parameter
        debug:      display debugging information
        ls_interp:  type of interpolation
        prog_tol:   minimum allowable step length
        do_plot:    do a graphical display of interpolation
        fun_obj:    objective function
        varargin:   parameters of objective function

    Outputs:
        t:          step length
        f_new:      function value at x + t * d
        g_new:      gradient value at x + t * d
        fun_evals:  number function evaluations performed by line search

    Remark: This function is a direct adaptation of Mark Schmidt's Matlab
    toolbox (minFunc) available at
    www.di.ens.fr/~mschmidt/Software/minFunc.html

    """

    if len(varargin) > 0:
        f_new, g_new = fun_obj(x + t * d, *varargin)
    else:
        f_new, g_new = fun_obj(x + t * d)
    f_prev, g_prev = (None, None)
    t_prev = None
    fun_evals = 1

    while f_new > fr + c1 * t * gtd or not is_legal(f_new):
        temp = t

        if ls_interp == 0 or not is_legal(f_new):
            if debug:
                print "Fixed BT"
            t *= .5
        elif ls_interp == 1 or not is_legal(g_new):
            #Use function value at new point, but not its derivative
            if fun_evals < 2 or ls_multi == 0 or not is_legal(f_prev):
                #Backtracking w/ quadratic interpolation based on two points
                if debug:
                    print "Quad BT"
                t = polyinterp(np.array([[0, f, gtd], [t, f_new, 1j]]),
                               do_plot,
                               0,
                               t)
            else:
                #Backtracking w/ cubic interpolation based on three points
                if debug:
                    print "Cubic BT"
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, 1j],
                                         [t_prev, f_prev, 1j]]),
                               do_plot,
                               0,
                               t)
        else:
            #Use function value and derivative at new point
            if fun_evals < 2 or ls_multi == 0 or not is_legal(f_prev):
                #Backtracking w/ cubic interpolation w/ derivative
                if debug:
                    print "Grad-Cubic BT"
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, g_new.T.dot(d)]]),
                               do_plot,
                               0,
                               t)
            elif not is_legal(g_prev):
                #Backtracking w/ quartic interpolation 3 points and derivative
                #of two
                if debug:
                    print "Grad-Quartic BT"
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, g_new.T.dot(d)],
                                         [t_prev, f_prev, 1j]]),
                               do_plot,
                               0,
                               t)
            else:
                #Backtracking w/ quintic interpolation of 3 points and
                #derivative of two
                if debug:
                    print "Grad-Quintic"
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, g_new.T.dot(d)],
                                         [t_prev, f_prev, g_prev.T.dot(d)]]),
                               do_plot,
                               0,
                               t)

        #Adjust if change in t is too small/large
        if t < temp * 1e-3:
            if debug:
                print "Interpolated Value Too Small, Adjusting"
            t = temp * 1e-3
        elif t > temp * .6:
            if debug:
                print "Interpolated Value Too Large, Adjusting"
            t = temp * .6

        #Store old point f doing three-point interpolation
        if ls_multi:
            f_prev = f_new
            t_prev = temp
            if ls_interp == 2:
                g_prev = g_new

        f_new, g_new = fun_obj(x + t * d, *varargin)
        fun_evals += 1

        #Check whether step size has become too small
        if np.max(np.abs(t * d)) <= prog_tol:
            if debug:
                print "Backtracking Line Search Failed"
            t = 0
            f_new = f
            g_new = g
            break

    x_new = x + t * d
    return (t, x_new, f_new, g_new, fun_evals)

def polyinterp(points, do_plot=0, xmin_bound=None, xmax_bound=None):
    """
    Return the minimum of interpolating polynomial.

    It can also be used for extrapolation if {xmin,xmax} are outside the
    domain of the points.

    Inputs:
        points(point_num, [x f g])
        do_plot:    set to 1 to plot, default: 0
        xmin:       min value that brackets minimum (default: min of points)
        xmax:       max value that brackets maximum (default: max of points)

    Set f or g to 1j if they are not known.
    The order of the polynomial is the number of known f and g values minus 1.

    Remark: This function is a direct adaptation of Mark Schmidt's Matlab
    toolbox (minFunc) available at
    www.di.ens.fr/~mschmidt/Software/minFunc.html

    """

    n_points = points.shape[0]
    order = np.sum(np.imag(points[:, 1:3]) == 0) - 1
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])

    #Compute bounds of interpolation area
    if xmin_bound is None:
        xmin_bound = xmin
    if xmax_bound is None:
        xmax_bound = xmax

    #Code for most common case: cubic interpolation of 2 points w/ function
    #and derivative values for both
    if n_points == 2 and order == 3 and do_plot == 0:
        #Solution in this case (where x2 is the farthest point):
        #   d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2);
        #   d2 = sqrt(d1^2 - g1 * g2);
        #   t_new = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2));
        #   min_pos = min(max(min_pos, x1), x2)
        min_val = np.min(points[:, 0])
        min_pos = np.nonzero(points[:, 0] == min_val)
        not_min_pos = -min_pos + 3

        g1 = points[min_pos, 2]
        g2 = points[not_min_pos, 2]
        f1 = points[min_pos, 1]
        f2 = points[not_min_pos, 2]
        x1 = points[min_pos, 0]
        x2 = points[not_min_pos, 0]

        d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
        d2 = np.sqrt(d1 ** 2 - g1 * g2)

        if np.isreal(d2):
            t = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            min_pos = min(max(t, xmin_bound), xmax_bound)
        else:
            min_pos = .5 * (xmax_bound + xmin_bound)

        return (min_pos, None)

    #Constraints based on available function values
    A = np.zeros((0, order + 1))
    b = np.zeros((0, 1))
    for i in xrange(n_points):
        if np.imag(points[i, 1]) == 0:
            constraint = np.zeros((order + 1))
            for j in xrange(order, -1, -1):
                constraint[order - j] = points[i, 0] ** j
            A = np.vstack((A, constraint))
            b = np.vstack((b, points[i, 1]))

    #Constraints based on available derivatives
    for i in xrange(n_points):
        if np.isreal(points[i, 2]):
            constraint = np.zeros((order + 1))
            for j in xrange(order):
                constraint[j] = (order - j) * points[i, 0] ** (order - j - 1)
            A = np.vstack((A, constraint))
            b = np.vstack((b, points[i, 2]))

    #Find interpolating polynomial
    params = np.linalg.lstsq(A, b)[0].flatten()

    #Compute critical points
    d_params = np.zeros((order))
    for i in xrange(len(params) - 1):
        d_params[i] = params[i] * (order - i)

    if np.isinf(d_params).any():
        cp = np.vstack((xmin_bound,
                        xmax_bound,
                        np.atleast_2d(points[:, 0]).T)).transpose()
    else:
        cp = np.vstack((xmin_bound,
                        xmax_bound,
                        np.atleast_2d(points[:, 0]).T,
                        np.atleast_2d(np.roots(d_params)).T)).transpose()

    #Test critical points
    fmin = np.inf
    #Default to bisection if no critical points valid
    min_pos = .5 * (xmin_bound + xmax_bound)
    for x_cp in cp.flatten():
        if np.imag(x_cp) == 0 and xmin_bound <= x_cp <= xmax_bound:
            f_cp = np.polyval(params, x_cp)
            if np.imag(f_cp) == 0 and f_cp < xmin:
                min_pos = np.real(x_cp)
                fmin = np.real(f_cp)

    #Plot Situation
    if do_plot:
        #TODO: manage plot with matplolib library
        pass

    return (min_pos, fmin)
