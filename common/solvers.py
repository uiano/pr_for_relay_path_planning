import numpy as np
try:
    import cvxopt as co
except ModuleNotFoundError:
    print('[solvers.py] CVXOPT not installed. ')
import scipy
from scipy.optimize import linprog

import logging

log = logging.getLogger("solvers")


def weighted_group_sparse_cvx(E, f, group_weights=None, **kwargs):
    """ This function solves the problem

        minimize_{y}  \sum_m |D_m^{-1} * y_m|_2 

        s.t.       E * y >= f

        where y = [y_1; y_2;...;y_{num_groups}] and each of the y_m has
        `num_vars_per_group` entries. 

        `group_weights` is given by [d_1, ... , d_{num_groups}], where diag(d_m) = D_m, and has
        num_vars_per_group rows. The entries should not be 0. 

        If `enforce_positivity==True`, then the constraint y>=0 is included.

        `f.ndim` must be 1.

        Returns:

            `Y_opt`: `num_vars_per_group` x `num_groups`
            `status` : str

        """
    num_vars_per_group = group_weights.shape[0]
    assert group_weights.shape[1] * group_weights.shape[0] == E.shape[1]

    Ep = E * np.ravel(group_weights.T)
    Y_opt, status = group_sparse_cvx(Ep, f, num_vars_per_group, **kwargs)
    Y_opt = Y_opt * group_weights

    return Y_opt, status


# def weighted_group_sparse_scipy_old(w, A, b, U):
#     """
#         minimize_{R,s}  w.T @ s

#         s.t. A @ vec(R) <= b

#             r_n <= s_n * ones(num_vars_per_group,1)

#             0 <= R <= U

#     where R = [r_1, r_2, ..., r_{num_groups}]

#     Inequalities hold entrywise.

#     See notes_cartography.pdf 2021/07/15

#     """
#     assert w.ndim == 1
#     assert len(w) == U.shape[1]
#     assert b.ndim == 1

#     num_groups = len(w)
#     num_vars_r = A.shape[1]
#     num_vars_per_group = U.shape[0]
#     num_constr = A.shape[0]

#     # x = [r; s]
#     c = np.concatenate((np.zeros((num_vars_r, )), w), axis=0)
#     A_ub_top = np.concatenate((A, np.zeros((num_constr, num_groups))), axis=1)
#     A_ub_bottom = np.concatenate(
#         [np.eye(num_vars_r), - np.repeat(np.eye(num_groups), num_vars_per_group, axis=0)], axis=1
#     )
#     A_ub = np.concatenate([A_ub_top,A_ub_bottom], axis=0)
#     b_ub = np.concatenate([b, np.zeros((num_vars_r,)) ], axis=0)
#     ub = np.concatenate(
#         [np.ravel(U.T),
#          np.full((num_groups, ), fill_value=None, dtype=float)],
#         axis=0)[:, None]
#     bounds = np.concatenate((np.zeros((num_vars_r + num_groups, 1)), ub),
#                             axis=1)
#     log.debug(f"The problem has {num_vars_r+num_groups} variables and {A_ub.shape[0]} non-box constraints.")
#     res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='revised simplex')
#     log.debug(f"fun. val = {res['fun']}")
#     x_opt = res.x
#     r_opt = x_opt[:num_vars_r]
#     R_opt = np.reshape(r_opt, (num_groups, num_vars_per_group)).T

#     status = "success" if res.success else "failure"

#     #log.debug("w=", w)
#     #log.debug("R_opt=", R_opt)
#     return R_opt, status


def weighted_group_sparse_scipy(w, A, b, U, thinning=True):
    """
        minimize_{R,s}  w.T @ s

        s.t. A @ vec(R) <= b

            r_n <= s_n * ones(num_vars_per_group,1)

            0 <= R <= U             

    where R = [r_1, r_2, ..., r_{num_groups}]

    Inequalities hold entrywise.

    If `thinning==True`, then the variables R_{mn} for which U_{mn} ==
    0 are removed. This reduces the number of variables used by the
    optimization algorithm. 

    See notes_cartography.pdf 2021/07/15

    """

    def apply_thinning(A_ub_top, A_ub_bottom, v_u, w):
        """Removes entries of the input matrices and vectors."""
        v_inds_r_remove = np.argwhere(v_u == 0)[:, 0]
        v_inds_r_keep = np.delete(np.arange(len(v_u)), v_inds_r_remove)

        log.debug(
            f"Removing {len(v_inds_r_remove)} out of {len(v_u)} variables.")

        v_u = np.delete(v_u, v_inds_r_remove, axis=0)
        A_ub_top = np.delete(A_ub_top, v_inds_r_remove, axis=1)
        A_ub_bottom = np.delete(A_ub_bottom, v_inds_r_remove, axis=1)
        A_ub_bottom = np.delete(A_ub_bottom, v_inds_r_remove, axis=0)

        # Remove also groups that completely disappeared --> remove columns from the right part of A_ub
        v_inds_s_remove = np.argwhere(np.sum(A_ub_bottom, axis=0) == 0)[:, 0]
        log.debug(f"{len(v_inds_s_remove)} groups have disappeared")
        assert np.sum(v_inds_s_remove < len(v_u)) == 0  # just in case
        A_ub_top = np.delete(A_ub_top, v_inds_s_remove, axis=1)
        A_ub_bottom = np.delete(A_ub_bottom, v_inds_s_remove, axis=1)
        w = np.delete(w, v_inds_s_remove - len(v_u), axis=0)

        return A_ub_top, A_ub_bottom, v_u, w, v_inds_r_keep

    def apply_thicking(r_opt_thin, v_inds):
        r_opt_thick = np.zeros((num_vars_r, ))
        r_opt_thick[v_inds] = r_opt_thin
        return r_opt_thick

    assert w.ndim == 1
    assert len(w) == U.shape[1]
    assert b.ndim == 1

    num_groups = len(w)
    num_vars_r = A.shape[1]
    num_vars_per_group = U.shape[0]
    num_constr = A.shape[0]

    # x = [r; s]

    A_ub_top = np.concatenate((A, np.zeros((num_constr, num_groups))), axis=1)
    A_ub_bottom = np.concatenate([
        np.eye(num_vars_r),
        -np.repeat(np.eye(num_groups), num_vars_per_group, axis=0)
    ],
                                 axis=1)
    v_u = np.ravel(U.T)

    log.debug(
        f"The problem has {num_vars_r+num_groups} variables and {A_ub_top.shape[0]+A_ub_bottom.shape[0]} non-box constraints."
    )
    if thinning:
        A_ub_top, A_ub_bottom, v_u, w, v_inds = apply_thinning(
            A_ub_top, A_ub_bottom, v_u, w)

    c = np.concatenate((np.zeros((len(v_u), )), w), axis=0)
    A_ub = np.concatenate([A_ub_top, A_ub_bottom], axis=0)
    b_ub = np.concatenate([b, np.zeros((A_ub_bottom.shape[0], ))], axis=0)
    bounds_top = np.concatenate([np.zeros((v_u.shape[0], 1)), v_u[:, None]],
                                axis=1)
    bounds_bottom = np.full((len(w), 2), fill_value=None, dtype=float)
    bounds = np.concatenate([bounds_top, bounds_bottom], axis=0)
    #bounds = np.concatenate((np.zeros((ub.shape[0], 1)), ub),                        axis=0)

    res = linprog(c,
                  A_ub=A_ub,
                  b_ub=b_ub,
                  bounds=bounds,
                  method='revised simplex')
    log.debug(f"fun. val = {res['fun']}")
    x_opt = res.x
    r_opt = x_opt[:len(v_u)]

    if thinning:
        r_opt = apply_thicking(r_opt, v_inds)
    R_opt = np.reshape(r_opt, (num_groups, num_vars_per_group)).T

    status = "success" if res.success else res["message"]

    #log.debug("w=", w)
    #log.debug("R_opt=", R_opt)
    return R_opt, status


def group_sparse_cvx(E,
                     f,
                     num_vars_per_group=None,
                     enforce_positivity=False,
                     method="primal",
                     group_tol=1e-4,
                     study_output=False):
    """ This function solves the problem
    
        minimize_{y}  \sum_m |y_m|_2 
    
        s.t.       E * y >= f

        where y = [y_1; y_2;...;y_{num_groups}] and each of the y_m has
        `num_vars_per_group` entries.

        If `enforce_positivity==True`, then the constraint y>=0 is included.
            
        `f.ndim` must be 1.

        Returns:

            `Y_opt`: `num_vars_per_group` x `num_groups`

            `status`: str

        """

    assert f.ndim == 1
    assert num_vars_per_group is not None
    num_vars_per_group = int(num_vars_per_group)
    num_vars = E.shape[1]
    assert num_vars % num_vars_per_group == 0
    num_groups = int(num_vars / num_vars_per_group)

    if enforce_positivity:
        E = np.concatenate((E, np.eye(num_groups * num_vars_per_group)),
                           axis=0)
        f = np.concatenate((f, np.zeros(num_groups * num_vars_per_group, )),
                           axis=0)

    num_constr = E.shape[0]

    # Let E = [E_1, E_2, ..., E_M]
    l_E = [
        E[:,
          ind_group * num_vars_per_group:(ind_group + 1) * num_vars_per_group]
        for ind_group in range(num_groups)
    ]

    def primal_mats():
        # Translation to CVX language (cf. notes-cartography.pdf)
        l_Gk = []
        l_hk = []
        for ind_group in range(num_groups):
            # These ones are G_1...G_M in CVX notation
            Gk = np.zeros((num_vars_per_group + 1,
                           num_groups * (num_vars_per_group + 1)))
            Gk[:, ind_group * (num_vars_per_group + 1):(ind_group + 1) *
               (num_vars_per_group + 1)] = -np.eye(num_vars_per_group + 1)
            hk = np.zeros((num_vars_per_group + 1, ))
            l_Gk.append(Gk)
            l_hk.append(hk)

        G0 = -np.concatenate([
            np.concatenate((np.zeros((num_constr, 1)), Ek), axis=1)
            for Ek in l_E
        ],
                             axis=1)
        h0 = -f

        c = np.zeros((num_groups * (num_vars_per_group + 1), ))
        inds = np.arange(0, len(c), num_vars_per_group + 1)
        c[inds] = 1
        return c, G0, h0, l_Gk, l_hk

    def dual_mats():
        # Translation to CVX language (cf. notes-cartography.pdf)
        l_Gk = []
        l_hk = []
        for ind_group in range(num_groups):
            # These ones are G_1...G_M in CVX notation
            Gk = -np.concatenate(
                (np.zeros((num_constr, 1)), l_E[ind_group]), axis=1).T
            hk = np.zeros((num_vars_per_group + 1, ))
            hk[0] = 1
            l_Gk.append(Gk)
            l_hk.append(hk)

        #G0 = np.concatenate((np.zeros((num_vars_per_group,1)),np.eye(num_vars_per_group)),axis=1).T
        G0 = np.eye(num_constr)

        h0 = np.zeros((num_constr, ))

        c = f
        return c, G0, h0, l_Gk, l_hk

    if method == "primal":
        c, G0, h0, l_Gk, l_hk = primal_mats()
    elif method == "dual":
        c, G0, h0, l_Gk, l_hk = dual_mats()
    else:
        raise ValueError

    def solve_and_check(feastol=1e-6):
        co.solvers.options['feastol'] = feastol
        p = co.solvers.socp(m(c), m(G0), m(h0), m(l_Gk), m(l_hk))

        if p['status'] != "optimal":
            log.debug("Status = ", p['status'])
            raise NotImplementedError
        else:
            status = 'success'

        # check feasibility
        x_opt = um(p['x'])[:, 0]
        s0 = um(p['sl'])[:, 0]
        slack_abserr = np.linalg.norm(G0 @ x_opt + s0 - h0)
        slack_relerr = slack_abserr / np.max((1, np.linalg.norm(h0)))
        log.debug(
            f"feastol = {feastol}, slack_abserr = {slack_abserr}, slack_relerr = {slack_relerr}"
        )

        zl = um(p['zl'])[:, 0]
        l_zq = [z[:, 0] for z in um(p['zq'])]
        z = np.concatenate([zl] + l_zq, axis=0)

        G = np.concatenate([G0] + l_Gk, axis=0)
        cvx_feas_dualtol = np.linalg.norm(G.T @ z + c) / np.max(
            (1, np.linalg.norm(c)))
        log.debug("cvx_dualfeas_tol = ", cvx_feas_dualtol)

        ##### debug
        # Gp = np.concatenate(l_Gk, axis=0)
        # zp = np.concatenate(l_zq, axis=0)
        # resp = - Gp.T@zp - c # must be entrywise nonnegative

        # Gpp = np.concatenate([Gk[1:,:] for Gk in l_Gk], axis=0)
        # zpp = np.concatenate([zq[1:] for zq in l_zq], axis=0)
        # respp = - Gpp.T@zpp - c # must be entrywise nonnegative

        # delta1 = - Gpp.T@zpp - E@zpp
        # delta2 = c - f
        # final = E@zpp -f
        # #####

        h = np.concatenate([h0] + l_hk, axis=0)
        sl = um(p['sl'])[:, 0]
        l_sq = [sq[:, 0] for sq in um(p['sq'])]
        s = np.concatenate([sl] + l_sq, axis=0)
        cvx_feas_primaltol = np.linalg.norm(G @ x_opt + s - h) / np.max(
            (1, np.linalg.norm(h)))
        log.debug("cvx_primalfeas_tol = ", cvx_feas_primaltol)
        return p, status, slack_abserr

    p, status, slack_err = solve_and_check(feastol=1e-7)

    # Output preparation
    if method == "primal":
        x_opt = um(p['x'])
        Y_opt = np.concatenate([
            x_opt[ind_group * (num_vars_per_group + 1) + 1:(ind_group + 1) *
                  (num_vars_per_group + 1)] for ind_group in range(num_groups)
        ],
                               axis=1)
    elif method == "dual":
        l_zq = [z[1:, 0:1] for z in um(p['zq'])]
        Y_opt = np.concatenate(l_zq, axis=1)
    else:
        raise ValueError

    def _study_output(E, f, Y_opt, status):
        if status != 'success':
            log.debug("No solution found")
            return

        np.set_printoptions(precision=2, linewidth=300)

        Y_opt_sp = group_sparsify(Y_opt, tol=group_tol)
        log.debug('[Sparsified solution; non-sparsified solution]:')
        log.debug(np.concatenate((Y_opt_sp, Y_opt), axis=0))

        norms = np.linalg.norm(Y_opt_sp, axis=0)
        log.debug(
            f"{len(norms[norms==0])} out of {len(norms)} groups are zero.")

        def study_feasibility(m_opt):
            v_opt = np.ravel(m_opt.T)
            res = E @ v_opt - f
            log.debug("   norm of the positive residuals (can be large):",
                      np.linalg.norm(res[res > 0]))
            log.debug("   norm of the negative residuals (should be small):",
                      np.linalg.norm(res[res < 0]))
            if enforce_positivity:
                log.debug(
                    "   fraction of the energy in the negative part of the solution:",
                    (np.linalg.norm(v_opt[v_opt < 0])**2) /
                    np.linalg.norm(v_opt)**2)

        log.debug("feasibility for the non-sparsified solution --------- ")
        study_feasibility(Y_opt)

        log.debug("feasibility for the sparsified solution --------- ")
        study_feasibility(Y_opt_sp)

        return

    if study_output:
        _study_output(E, f, Y_opt, status)

    return Y_opt, status


def m(A):
    if isinstance(A, list):
        return [m(Am) for Am in A]
    return co.matrix(A)


def um(M):  #"unmatrix"
    if isinstance(M, list):
        return [um(Mm) for Mm in M]
    return np.array(M)


def sparsify(M, tol=0.01):
    n = np.linalg.norm(np.ravel(M), ord=1)
    M = np.copy(M)
    M[M < tol * n] = 0
    return M


def group_sparsify(M, tol=0.01):
    n = np.max(np.max(M, axis=0))
    M = np.copy(M)
    for ind_col in range(M.shape[1]):
        if np.max(M[:, ind_col]) < tol * n:
            M[:, ind_col] = 0
    return M
