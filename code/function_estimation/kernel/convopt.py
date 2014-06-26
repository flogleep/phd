from    backtracking    import  backtracking
import  numpy           as      np
import  pdb

def minimize_f(x0, f, grad_f, bounds = None, method = 'projected_gradient'):
    """Solution of convex optimization problem, possibly with constraints.

    Inputs:
        x0:     Starting point
        f:      Objective function
        grad_f: Gradient of f
        bounds: Upper and lower bounds of the optimization problem. If
                specified, it must be of the form [(lower_b, upper_b)] and its
                length must be the same as the dimension of the problem
        method: Method used for solving the optimization problem

    Outputs:
        x:      Solution of the optimization problem
        f_x:    Value of the function at x
        grad_x: Gradient at x
        it:     Number of iterations needed
    """

    x           = x0.copy()
    fun_obj     = lambda x: (f(x), grad_f(x))
    f_x, grad_x = fun_obj(x)
    c1          = 1e-1
    ls_interp   = 1
    ls_multi    = 1
    convergence = False
    precision   = 1e-5
    it          = 0
    it_max      = 5000
    if bounds == None:
        l_bounds = -np.inf * np.ones(x.shape)
        u_bounds = np.inf * np.ones(x.shape)
    else:
        l_bounds = np.array([l for (l,u) in bounds])
        u_bounds = np.array([u for (l,u) in bounds])

    while not convergence:
        it += 1

        d   = -grad_x.copy()
        gtd = np.inner(d, grad_x)
        t   = 1

        t, x_new, f_new, g_new, _ = backtracking(x, t, d, f_x, grad_x, f_x, \
                gtd, c1, ls_interp = ls_interp, ls_multi = ls_multi, \
                fun_obj = fun_obj)

        x   = x_new.copy()
        x   = np.maximum(x, l_bounds)
        x   = np.minimum(x, u_bounds)
        if (x != x_new).any():
            f_new = f(x)
            g_new = grad_f(x)

        convergence = np.inner(f_new - f_x, f_new - f_x) < precision ** 2 \
                and it < it_max

        f_x     = f_new.copy()
        grad_x  = g_new.copy()

    return (x, f_x, grad_x, it)
