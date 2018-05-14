#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import numpy as np
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

def solve_lasso(V, Y, constraint_ub, gamma_init=None):
    """
    Solves the problem
    min_{U} || U ||_1
    s.t.    (1/2) * || y_i - V*u_i ||_2^2 <= T_i, i = 1, 2, \dots, p
    using successive solutions of the Lagrangian problem for varying values of
    the lagrange multiplier, until the complementary slackness KKT condition is
    satisfied for each constraint. The Lagrangian problem splits into p
    individual vector LASSO problems, which we solve with ADMM. We search over
    dual variables until complementary slackness is satisfied using binary
    search. Tolerances can be set internally:
    - TOL_SOLVE: ADMM quits once the residual norm squared is below this value
    - TOL_BSEARCH: Binary search quits once the interval size is below this
       value

    For reference, concerning inputs and their sizes:
    Y is p by m
    U is p by n (not an input, but involved in the problem above)
    V is m by n
    constraint_ub is p by 1 vector (Constraints for each row)
    gamma_init is p by 1 vector (Guesses for upper bound on each dual variable
             search interval: estimate using knowledge of noise powers?)

    TODO:
    1. Expose tolerances?

    """
    # Data dimensions
    (p, m) = Y.shape
    n = V.shape[1]
    # Assert checks (data dimensions)
    assert V.shape[0] == m
    assert constraint_ub.shape[0] == p
    assert gamma_init.shape[0] == p

    # Solver parameters
    MAX_ITER_INNER = int(1e3)
    MAX_ITER_OUTER = int(1e2)
    TOL_SOLVE = 1e-8
    TOL_BSEARCH = 1e-4
    mu = 1e0 * 1.0
    gamma = np.zeros([p, 2])
    gamma[:, 1] = gamma_init

    # objective tracking
    slack = np.nan * np.ones([MAX_ITER_OUTER, 1])
    res = np.nan * np.ones([MAX_ITER_INNER, 1])

    # Cache variables for initialization / warm starting
    X = np.zeros([p, n])
    Z = np.zeros([p, n])
    D = np.zeros([p, n])
    VtV = np.dot(V.T, V)
    VtY = np.dot(V.T, Y.T)

    # Solver main loop
    for row in range(p):
        # Outer loop is over rows (dual variables): for each row in the image

        row_gamma = np.mean(gamma[row, :])
        y = Y[row, :].T
        Vty = VtY[:, row]
        # initialize / warm start
        x = X[row, :].T
        z = Z[row, :].T
        d = D[row, :].T
        for iter_outer in range(MAX_ITER_OUTER):
            # This loop finds the optimal dual variable for this row using
            # binary search

            AtA = row_gamma/mu * VtV + sp.sparse.spdiags(np.ones([n]), 0, n, n)
            L = np.linalg.cholesky(AtA)
            Aty = (row_gamma / mu * Vty)
            for iter_inner in range(MAX_ITER_INNER):
                # This loop repeatedly solves LASSO problems with ADMM for a
                # given value of the dual variable

                # X step
                x = sf(z - d / mu, 1/mu)
                # Z step
                tmp = np.linalg.solve(L, Aty + x + d/mu)
                z = np.linalg.solve(L.T, tmp)
                # DUAL ASCENT (D step)
                d = d + mu * (x - z)
                # Tracking
                res[iter_inner] = np.linalg.norm(x - z, ord=2)**2

                # Exit checks
                if iter_inner > 1 and np.abs(res[iter_inner] - res[iter_inner-1]) <= TOL_SOLVE:
                    break

            if iter_inner == MAX_ITER_INNER - 1:
                print("ADMM subproblem solver failed to converge")
            # Subproblem solved: update lambda and keep searching
            slack[iter_outer] = np.linalg.norm(y - np.dot(V, x.T), ord=2)**2 - constraint_ub[row]
            if slack[iter_outer] < 0:
                # Constraint satisfied, but not tight: decrease gamma
                gamma[row, 1] = np.mean(gamma[row, :])
            else:
                # Constraint violated: increase lambda
                gamma[row, 0] = np.mean(gamma[row, :])

            # Update the mean value of gamma and reloop
            row_gamma = np.mean(gamma[row, :])

            # Check if we should stop bsearching
            if np.abs(gamma[row, 1] - gamma[row, 0]) <= TOL_BSEARCH:
                print("Finished binary search for row %d at iter %d" % (row,
                                                                        iter_outer))
                break

        # Subproblem solved: update outputs
        X[row, :] = x
        Z[row, :] = z
        D[row, :] = d

    # Solved every row: return and exit
    return X


def sf(x, gamma):
    """
    Soft thresholding with parameter gamma
    """
    y = np.max(np.vstack((np.zeros_like(x), x - gamma)), axis=0) - np.max(np.vstack((np.zeros_like(x), -x - gamma)), axis=0)
    return y


if __name__ == "__main__":
    # Some test data, with a plot
    m = 1000
    n = 5
    p = 100
    k = 50
    noise_std = 1e-2

    V = la.orth(np.random.randn(m, n)) + np.sqrt(1e-2) * np.random.randn(m,n)
    idxs = np.random.choice(range(n*p), k, replace=False)
    Uv = sp.sparse.coo_matrix((np.random.randn(k), (idxs, np.ones([k]))),
                              shape=(n*p,2))
    Uv = Uv.todense()
    Uv = Uv[:, 1]
    U = np.reshape(Uv, [p, n])
    U = np.asarray(U)
    noise = noise_std * np.random.randn(p, m)
    Y = np.dot(U, V.T) + noise
    T = 2 * np.linalg.norm(noise, 2)**2
    X = solve_lasso(V, Y, T/p * np.ones([p]), 1e4 * np.ones([p]))

    plt.stem(X.reshape(np.prod(X.shape)))
    plt.stem(U.reshape(np.prod(U.shape)), "rx")
    plt.show()
