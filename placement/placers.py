from cmath import inf
from filecmp import DEFAULT_IGNORES
from itertools import count
from sre_constants import SUCCESS
import sys
import os
from tabnanny import check
from IPython.core.debugger import set_trace
from collections import OrderedDict

import numpy as np

from numpy import maximum as np_maximum
from numpy import minimum as np_minimum
from numpy import expand_dims as np_expand_dims
from numpy import sum as np_sum

import matplotlib.pyplot as plt
from numpy.core.defchararray import endswith
from sklearn.cluster import KMeans
import scipy
from scipy.optimize import linprog, bisect

from gsim import rng
from gsim.utils import xor

from common.fields import FunctionVectorField, VectorField
from common.grid import RectangularGrid3D
from common.solvers import group_sparse_cvx, sparsify, group_sparsify, weighted_group_sparse_cvx, weighted_group_sparse_scipy
from common.runner import Runner
import warnings
import logging
import pickle

import cProfile

log = logging.getLogger("placers")


class FlyGrid(RectangularGrid3D):

    def __init__(self,
                 *args,
                 f_disable_indicator=None,
                 min_height=None,
                 **kwargs):
        """ Args: `f_disable_indicator` is a function that takes a vector
            `coords` of shape (3,) with the coordinates of a point and returns a
            vector or scalar. A grid point with coordinates `coords` and height
            >= `min_height` is enabled (flying allowed) iff
            `f_disable_indicator(coords)` is False, 0, or a 0 vector. 
        """
        super().__init__(*args, **kwargs)

        if f_disable_indicator is not None:
            self.disable_by_indicator(f_disable_indicator)

        self._min_height = min_height
        if self._min_height is not None:
            self.disable_by_indicator(lambda coords:
                                      (coords[2] < self._min_height))

    @property
    def min_height(self):
        return self._min_height

    # def plot_as_blocks(self):
    # #     return self.enable_field.plot_as_blocks()

    # def plot_allowed_pts(self, ax):
    #     """ `ax` is a pyplot 3D axis"""
    #     allowed_pts = self.list_pts()
    #     ax.plot(allowed_pts[:, 0], allowed_pts[:, 1], allowed_pts[:, 2], ".")


class Placer():
    """ Abstract class for Placers"""

    # to be overridden by subclasses
    _name_on_figs = ""

    # This is set at construction time and should not be modified by subclasses
    def __init__(self):
        pass

    # To be implemented by subclasses
    def place(self, fly_grid, channel, user_coords):
        """
        Args: 

        - `user_coords` is num_users x 3

        Returns:

        - `uav_coords` is num_uav x 3

       
        """
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def name_on_figs(self):
        if self._name_on_figs:
            return self._name_on_figs
        else:
            return self.name


class CapacityBasedPlacer(Placer):

    def __init__(self, min_user_rate=None, max_uav_total_rate=None, **kwargs):
        super().__init__(**kwargs)

        self.min_user_rate = min_user_rate
        self.max_uav_total_rate = max_uav_total_rate

    def plot_capacity_maps(self, fly_grid, channel, user_coords, znum=4):
        """ Returns one GFigure for each user with the corresponding capacity map."""
        maps = channel.capacity_maps(
            grid=fly_grid,
            user_coords=user_coords,
        )
        return [map.plot_z_slices(znum=4) for map in maps]

    # def get_capacity_map(self, user_coords):
    #     return

    @staticmethod
    def find_rate_allocation(m_capacity,
                             min_user_rate,
                             max_uav_total_rate,
                             v_real_uav_inds=None):
        """
            Input:

                If `v_real_uav_inds` is None, then `m_capacity` is num_users x
                num_uavs. 

                If `v_real_uav_inds` is a length-num_uavs vector of indices of
                grid points, then `m_capacity` must be num_users x
                num_grid_points. 

            Returns:

                `m_rate`: None if a feasible allocation does not exist. Else,
                this is a matrix of the same shape as `m_capacity`. If
                `v_real_uav_inds` is not None, then the columns of `m_rate`
                whose index is not in `v_real_uav_inds` will be 0.


        """

        if v_real_uav_inds is None:
            return CapacityBasedPlacer.find_rate_allocation_real_uavs(
                m_capacity, min_user_rate, max_uav_total_rate)
        else:
            m_capacity_submat = m_capacity[:, v_real_uav_inds]

            m_rate_submat = CapacityBasedPlacer.find_rate_allocation_real_uavs(
                m_capacity_submat, min_user_rate, max_uav_total_rate)

            if m_rate_submat is None:
                return None

            m_rate = np.zeros(m_capacity.shape)
            for ind_uav in range(len(v_real_uav_inds)):
                m_rate[:, v_real_uav_inds[ind_uav]] = m_rate_submat[:, ind_uav]

            return m_rate

    @staticmethod
    def find_rate_allocation_real_uavs(m_capacity_submat, min_user_rate,
                                       max_uav_total_rate):
        """

            Input:
                `m_capacity_submat` is num_users x num_uavs.

            Returns:
                If a feasible rate allocation exists, it returns a num_users x
                num_uavs matrix that satisfies all constraints. Else, it returns
                None.
        
        """

        num_users, num_uavs = m_capacity_submat.shape
        num_variables = m_capacity_submat.size

        c = np.zeros(num_variables)
        A_ub = np.repeat(np.eye(num_uavs), num_users, axis=1)

        assert max_uav_total_rate != None

        b_ub = max_uav_total_rate * np.ones(num_uavs)
        A_eq = np.tile(np.eye(num_users), num_uavs)
        b_eq = min_user_rate * np.ones(num_users)

        v_sub_capacity = np.expand_dims(m_capacity_submat.flatten("F"), 1)
        v_zeros = np.zeros((v_sub_capacity.size, 1))
        m_bound = np.concatenate((v_zeros, v_sub_capacity), 1)

        res = linprog(c,
                      A_ub=A_ub,
                      b_ub=b_ub,
                      A_eq=A_eq,
                      b_eq=b_eq,
                      bounds=m_bound,
                      method='revised simplex')

        if res.success:
            return np.reshape(res.x, (num_users, num_uavs), order='F')
        else:
            return None

    # @staticmethod
    # def find_association(m_capacity_submat,
    #                      min_user_rate,
    #                      max_uav_total_rate,
    #                      mode='single'):
    #     """

    #         Input:
    #             `m_capacity_submat` is num_users x num_uavs.

    #         Returns:
    #             If a feasible rate allocation exists, it returns a num_users x
    #             num_uavs matrix that satisfies all constraints. Else, it returns
    #             None.

    #     """

    #     num_users, num_uavs = m_capacity_submat.shape

    #     if mode=='single':
    #         return
    #     else
    #         return


class SingleUAVPlacer(CapacityBasedPlacer):

    def __init__(self, criterion="sum_rate", num_uavs=1, **kwargs):

        assert num_uavs == 1
        super().__init__(**kwargs)

        self.criterion = criterion

    def place(self, fly_grid, channel, user_coords=None):
        """
        See parent.
        """
        # TODO: change the following line to use channel.capacity_map. This
        # would unify GroupSparseUAVPlacer.place and SingleUAVPlacer.place but
        # requires the rest of this function.
        maps = channel.capacity_maps(grid=fly_grid, user_coords=user_coords)

        if self.criterion == "sum_rate":
            sum_rate_map = VectorField.sum_fields(maps)
            debug = 0
            if debug:
                F = sum_rate_map.plot_z_slices(znum=4)
                F.plot()
                plt.show()

            uav_coords = sum_rate_map.arg_coord_max()
        elif self.criterion == "max_min_rate":
            # maximize the minimum rate across grid points
            min_rate_map = VectorField.min_over_fields(maps)
            uav_coords = min_rate_map.arg_coord_max()
        else:
            raise ValueError("unrecognized self.criterion = ", self.criterion)

        return np.array([uav_coords])


class SparseUAVPlacer(CapacityBasedPlacer):
    """
    A subset of grid pts is selected by solving:

    minimize_{v_alpha}    || v_alpha ||_1

    s.t.                  m_C @ v_alpha >= min_user_rate * ones(num_users,)

                          0 <= v_alpha <= 1

    where m_C is num_users x num_pts and contains the capacity of each link.
    Those points associated with a positive entry of the optimal v_alpha
    indicate the positions of the UAVs.

    """
    _name_on_figs = "Sparse Placer (proposed)"

    def __init__(self,
                 sparsity_tol=1e-2,
                 num_max_reweight_iter=4,
                 epsilon_reweighting=1e-2,
                 **kwargs):
        """ Args:

            `sparsity_tol`: tolerance for sparsifying the groups. 

            `num_max_reweight_iter`: maximum number of times that reweighting is 
            applied when solving the placement opt. problem.
        """

        super().__init__(**kwargs)
        assert self.max_uav_total_rate is None, "this placer cannot guarantee a max total rate per UAV"

        self.sparsity_tol = sparsity_tol
        self.num_max_reweight_iter = num_max_reweight_iter
        self.epsilon_reweighting = epsilon_reweighting

    def place(self, fly_grid, channel, user_coords=None):
        """
        See parent.
        """

        #     max_rate_map = VectorField.clip(map, upper=self.min_user_rate)
        #     max_rate_map.disable_gridpts_by_dominated_verticals()

        #     return self._sparse_placement(max_rate_map)

        # def _sparse_placement(self, map):

        def get_pt_soft_indicators(gridpt_weights, m_capacity):

            #gridpt_weights = gridpt_weights/np.sum(gridpt_weights)
            bounds = np.zeros((num_pts, 2))
            bounds[:, 1] = 1
            A_ub = -m_capacity
            b_ub = -self.min_user_rate * np.ones((num_users, ))
            with warnings.catch_warnings():
                # Suppress deprecation warning inside SciPy
                warnings.simplefilter("ignore")

                num_failures = 0
                while True:
                    res = linprog(gridpt_weights,
                                  A_ub=A_ub,
                                  b_ub=b_ub,
                                  bounds=bounds)
                    v_weights = res.x
                    if any(np.isnan(v_weights)):
                        log.warning(
                            "SparseUAVPlacer: Linprog returned a NaN solution")
                        gridpt_weights = 0.1 * gridpt_weights
                        num_failures += 1
                        if num_failures >= 5:
                            set_trace()
                            print("linprog failed Nan")

                    else:
                        break
            status = "success" if res.success else res["message"]

            #debug
            # res2= linprog(gridpt_weights/np.sum(gridpt_weights),
            #                   A_ub=A_ub,
            #                   b_ub=b_ub,
            #                   bounds=bounds)
            # v_weights2 = res2.x
            # diff = np.linalg.norm(v_weights-v_weights2)
            # print("diff = ", diff)
            # if diff>0.1:
            #     print("different")

            return v_weights, status

        def weights_to_rates_and_coords(v_pt_weights, m_capacity):
            v_sp_pt_weights = sparsify(v_pt_weights, self.sparsity_tol)
            v_inds = np.nonzero(v_sp_pt_weights)[0]
            m_rates = m_capacity[:, v_inds]
            uav_coords = map.grid.list_pts()[v_inds, :]
            return uav_coords, m_rates

        np.set_printoptions(precision=2, suppress=True, linewidth=2000)

        map = channel.capacity_map(
            grid=fly_grid,
            user_coords=user_coords,
        )
        m_capacity = map.list_vals().T
        # The following improves the results
        m_capacity = np.minimum(m_capacity, self.min_user_rate)
        num_users, num_pts = m_capacity.shape

        # Reweighting [Candes, Wakin, and Boyd, 2008]
        # TODO: experiment with the initial weights
        gridpt_weights = 1 / (1 + np.sum(m_capacity >= self.min_user_rate) +
                              1e-4 * np.random.random((num_pts, )))
        num_uavs_prev = None
        for ind in range(self.num_max_reweight_iter):
            v_pt_soft_indicators, status = get_pt_soft_indicators(
                gridpt_weights, m_capacity)
            m_uav_coords, m_rates = weights_to_rates_and_coords(
                v_pt_soft_indicators, m_capacity)
            num_uavs = m_rates.shape[1]

            log.debug(f"Number of UAVs after {ind+1} iterations = {num_uavs}")
            log.debug("UAV rates=")
            log.debug(m_rates)

            if (num_uavs == 1) or (num_uavs == num_uavs_prev):
                break
            num_uavs_prev = num_uavs

            gridpt_weights = 1 / (v_pt_soft_indicators +
                                  self.epsilon_reweighting)

        return m_uav_coords


class GroupSparseUAVPlacer(CapacityBasedPlacer):

    def __init__(self,
                 criterion="min_uav_num",
                 sparsity_tol=1e-2,
                 backend="scipy",
                 admm_stepsize=1e-8,
                 admm_max_num_iter=300,
                 admm_feasibility_tol=10,
                 admm_initial_error_tol=10,
                 reweighting_noise=0.1,
                 reweighting_num_iter=1,
                 epsilon_reweighting=1e-2,
                 normalize=True,
                 eps_abs=1e-5,
                 eps_rel=1e-5,
                 b_admm_decrease_err_tol=True,
                 b_plot_progress=False,
                 b_save_group_weights=False,
                 b_load_group_weights=False,
                 **kwargs):
        """ Args:
            `group_tol`: tolerance for sparsifying the groups.
        """

        super().__init__(**kwargs)
        assert self.max_uav_total_rate is not None
        self.criterion = criterion
        self.sparsity_tol = sparsity_tol
        self.backend = backend
        self.admm_stepsize = admm_stepsize
        self.admm_max_num_iter = admm_max_num_iter
        self.reweighting_noise = reweighting_noise
        self.reweighting_num_iter = reweighting_num_iter
        self.epsilon_reweighting = epsilon_reweighting
        self.normalize = normalize
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.b_admm_decrease_err_tol = b_admm_decrease_err_tol
        self.b_plot_progress = b_plot_progress
        self.b_save_group_weights = b_save_group_weights
        self.b_load_group_weights = b_load_group_weights
        self.admm_feasibility_tol = admm_feasibility_tol
        self.admm_initial_error_tol = admm_initial_error_tol

    def place(self, fly_grid, channel, user_coords=None):
        """
        See parent.
        """

        map = channel.capacity_map(
            grid=fly_grid,
            user_coords=user_coords,
        )

        if self.criterion == "min_uav_num":

            prof = cProfile.Profile()
            prof.enable()

            uav_coords = self._min_uav_num_placement(map)

            prof.disable()
            prof.dump_stats('output/profiled_admm.prof')

        else:
            raise ValueError("unrecognized self.criterion = ", self.criterion)

        return uav_coords

    def _min_uav_num_placement(self, map):
        # def infeasible(m_capacity):
        #     # m_C = np.copy(m_capacity)
        #     # if self.min_link_capacity is not None:
        #     #     m_C[m_C < self.min_link_capacity] = 0
        #     s = np.sum(m_capacity, axis=1)
        #     coverable_users = (s >= self.min_user_rate)
        #     return np.sum(coverable_users) < len(s)

        np.set_printoptions(precision=2, suppress=True, linewidth=2000)

        # The following is num_users x num_points. Each column is a group.
        m_capacity = map.list_vals().T

        num_users, num_pts = m_capacity.shape

        if self.normalize:
            m_capacity = m_capacity / (num_users * num_pts)
            self.max_uav_total_rate = self.max_uav_total_rate / (num_users *
                                                                 num_pts)
            self.min_user_rate = self.min_user_rate / (num_users * num_pts)

        # if infeasible(m_capacity):
        #     raise NotImplementedError
        # else:
        #     log.debug("The problem may be feasible.")

        if self.backend == "cvx_matlab":
            m_uav_rates, status = self._min_uav_num_placement_cvx_matlab(
                m_capacity)
        elif self.backend == "cvx_python":
            m_uav_rates, status = self._min_uav_num_placement_cvx_python(
                m_capacity)
        elif self.backend == "scipy":

            m_uav_rates, status = self._min_uav_num_placement_scipy_reweight(
                m_capacity)

        elif self.backend == "admm":

            m_uav_rates, status = self._min_uav_num_placement_admm(m_capacity)

        else:
            raise ValueError

        # # # print optimal values and infeasibility for m_uav_rates
        # group_weights = 1 / (1 + np.sum(m_capacity >= self.min_user_rate) +
        #                      self.reweighting_noise * np.random.random((m_capacity.shape[1], )))
        # # group_weights = np.ones(num_pts) + 1/np.linspace(1,num_pts,num_pts)   # this one is for checking the feasibility
        # objective, total_infeasibility = self._feasibility_check(group_weights, m_capacity, m_uav_rates)
        # print("Objective: {}, total infeasibility: {}".format(objective, total_infeasibility))
        # print("m_rates: ", m_uav_rates)
        # v_inf_norms = np.linalg.norm(m_uav_rates, axis=0, ord=np.inf)
        # print("inf norms: ", v_inf_norms)

        if status != "success":
            log.debug("Optimization failed:", status)
            raise NotImplementedError

        if self.normalize:
            # unnormalize
            self.max_uav_total_rate = self.max_uav_total_rate * (num_users *
                                                                 num_pts)
            self.min_user_rate = self.min_user_rate * (num_users * num_pts)

            m_uav_rates = m_uav_rates * num_users * num_pts

        #v_uav_rates = np.ravel(m_uav_rates.T)

        uav_coords, uav_rates, uav_upper_capacity_limit = self._uav_rates_to_coords(
            m_uav_rates, map, group_tol=self.sparsity_tol)

        # log.debug("uav_coords = \n", uav_coords)
        log.debug("uav rates for each user = \n", uav_rates)
        # log.debug("uav_upper_capacity_limit = \n", uav_upper_capacity_limit)
        log.debug("total uav rates = \n", np.sum(uav_rates, axis=0))
        # log.debug("total user rates = \n", np.sum(uav_rates, axis=1))
        # #log.debug("objective = \n", np.sum(np.linalg.norm(uav_rates, axis=1)))
        log.debug("objective = \n",
                  np.sum(np.linalg.norm(uav_rates, axis=0, ord=np.inf)))
        # print("objective = \n",
        #           np.sum(np.linalg.norm(uav_rates, axis=0, ord=np.inf)))

        return uav_coords  #, uav_rates, uav_upper_capacity_limit  # no refining

        uav_coords, uav_rates, uav_upper_capacity_limit = self.refine(
            uav_coords, uav_rates, uav_upper_capacity_limit)
        log.debug("After refining---------------")
        log.debug("uav_coords = \n", uav_coords)
        log.debug("uav rates for each user = \n", uav_rates)
        log.debug("uav_upper_capacity_limit = \n", uav_upper_capacity_limit)
        log.debug("total uav rates = \n", np.sum(uav_rates, axis=0))
        log.debug("total user rates = \n", np.sum(uav_rates, axis=1))
        #log.debug("objective = \n", np.sum(np.linalg.norm(uav_rates, axis=1)))
        log.debug("objective = \n",
                  np.sum(np.linalg.norm(uav_rates, axis=0, ord=np.inf)))
        return uav_coords  #, uav_rates, uav_upper_capacity_limit

    def _uav_rates_to_coords(self, m_uav_rates, map, group_tol):

        m_pts = map.grid.list_pts()
        if m_uav_rates.shape[1] != m_pts.shape[0]:
            raise ValueError

        m_uav_rates_sp = group_sparsify(m_uav_rates, tol=group_tol)
        v_inds = np.nonzero(np.linalg.norm(m_uav_rates_sp, axis=0))[0]
        uav_coords = m_pts[v_inds, :]
        uav_rates = m_uav_rates_sp[:, v_inds]
        uav_upper_capacity_limit = map.list_vals()[v_inds, :].T
        return uav_coords, uav_rates, uav_upper_capacity_limit

    # not currently used
    def refine(self, uav_coords, uav_rates, uav_upper_capacity_limit):

        R_bar = np.minimum(uav_upper_capacity_limit, self.min_user_rate)
        num_pts = uav_rates.shape[1]
        num_users = uav_rates.shape[0]
        bounds = np.concatenate((np.zeros((num_pts, 1)), np.ones(
            (num_pts, 1))),
                                axis=1)
        #c = np.ones((num_pts,))
        c = 1 + 1e-3 * np.random.random((num_pts, ))
        res = linprog(c,
                      A_ub=-R_bar,
                      b_ub=-self.min_user_rate * np.ones((num_users, )),
                      bounds=bounds)

        alpha_opt = res.x
        alpha_opt_sp = group_sparsify(alpha_opt[None, :])
        v_inds = np.nonzero(alpha_opt_sp[0, :])[0]
        uav_coords = uav_coords[v_inds, :]
        uav_rates = R_bar[:, v_inds]
        uav_upper_capacity_limit = uav_upper_capacity_limit[:, v_inds]
        return uav_coords, uav_rates, uav_upper_capacity_limit

    def _min_uav_num_placement_cvx_matrices(
            self,
            m_capacity,
            min_rate_scale=1  # Weight of the constraint (force feasibility...)
    ):

        num_users, num_pts = m_capacity.shape

        E = np.concatenate((
            np.tile(min_rate_scale * np.eye(num_users), reps=(1, num_pts)),
            -np.repeat(np.eye(num_pts), num_users, axis=1),
            np.eye(num_users * num_pts),
            -np.eye(num_users * num_pts),
        ),
                           axis=0)

        f = np.concatenate(
            (np.tile([[min_rate_scale * self.min_user_rate]],
                     (num_users, 1)), -np.tile([[self.max_rate]],
                                               (num_pts, 1)),
             np.zeros((num_users * num_pts, 1)),
             -np.ravel(m_capacity.T)[:, None]))[:, 0]

        return E, f

    def _min_uav_num_placement_cvx_matlab(self, m_capacity):
        num_users, num_pts = m_capacity.shape

        E, f = self._min_uav_num_placement_cvx_matrices(m_capacity)

        r = Runner("placement", "sparsity.m")
        data_in = OrderedDict()
        data_in["E"] = E
        data_in["f"] = f
        data_in["num_vars_per_group"] = num_users
        data_in["C"] = m_capacity

        # Option 1:
        #r.run("save_data", data_in)
        # Process the data in MATLAB afterwards
        #
        # Option 2: assignment from matlab
        m_uav_rates = r.run("sparse_placement", data_in)[0]
        status = 'failure' if np.isnan(m_uav_rates[0, 0]) else 'success'
        return m_uav_rates, status

    def _min_uav_num_placement_cvx_python(self, m_capacity):
        num_users, num_pts = m_capacity.shape

        E, f = self._min_uav_num_placement_cvx_matrices(m_capacity)

        # TODO: decide which set o
        # group_weights = 1 / (1 + 1e-4 * np.random.random(
        #     (num_users, num_pts)))
        #group_weights = np.tile( 1 + 1/(np.sum(m_capacity,axis=0)), reps=(num_users,1))
        group_weights = np.tile(
            1 + 1 /
            (np.sum(np.minimum(m_capacity, self.min_user_rate), axis=0)),
            reps=(num_users, 1))
        m_uav_rates, status = weighted_group_sparse_cvx(
            E,
            f,
            group_weights=group_weights,
            enforce_positivity=False,
            method="dual",
            group_tol=self.sparsity_tol,
            study_output=True)

        return m_uav_rates, status

    def _min_uav_num_placement_scipy(self, m_capacity):

        num_users, num_pts = m_capacity.shape

        A = -np.tile(np.eye(num_users), reps=(1, num_pts))
        #b = -np.tile([[self.min_user_rate]], (num_users,1 ))
        b = -np.tile([self.min_user_rate], (num_users, ))
        if self.max_uav_total_rate is not None:
            A = np.concatenate((
                A,
                np.repeat(np.eye(num_pts), num_users, axis=1),
            ),
                               axis=0)
            b = np.concatenate([
                b[:, None],
                np.tile([[self.max_uav_total_rate]], (num_pts, 1))
            ],
                               axis=0)[:, 0]
        U = m_capacity

        # m_uav_rates, status = self._loop(m_capacity, lambda group_weights:weighted_group_sparse_scipy(
        #         group_weights, A, b, U))

        # TODO: experiment with the initial weights
        group_weights = 1 / (1 + np.sum(m_capacity >= self.min_user_rate) +
                             self.reweighting_noise * np.random.random(
                                 (num_pts, )))
        # testing purpose
        # group_weights = np.ones(m_capacity.shape[1])
        # group_weights = np.ones(num_pts) + 1/np.linspace(1,num_pts,num_pts)

        # ########## test
        # m_uav_rates, status = weighted_group_sparse_scipy(
        #     group_weights, A, b, U, min_link_capacity=self.min_link_capacity)
        # U[U < self.min_link_capacity] = 0
        # m_uav_rates_old, status = weighted_group_sparse_scipy_old(
        #     group_weights, A, b, U)
        # log.debug("difff=", np.linalg.norm(m_uav_rates - m_uav_rates_old))
        #############
        #log.debug("m_uav_rates=", m_uav_rates)

        # Reweighting [Candes, Wakin, and Boyd, 2008]
        num_uavs_prev = None
        for ind in range(self.reweighting_num_iter):
            # log.debug(
            #     "delta=",
            #     np.linalg.norm(1 / (np.sum(m_uav_rates, axis=0) + epsilon) -
            #                    group_weights))

            m_uav_rates, status = weighted_group_sparse_scipy(
                group_weights, A, b, U)
            num_uavs = np.sum(
                np.sum(group_sparsify(m_uav_rates, self.sparsity_tol), axis=0)
                > 0)
            log.debug(f"Number of UAVs after {ind+1} iterations = {num_uavs}")

            if (num_uavs == 1) or (num_uavs == num_uavs_prev):
                break
            num_uavs_prev = num_uavs
            #log.debug("m_uav_rates=", m_uav_rates)
            #log.debug("wg = ", group_weights)
            #nothing = 0
            group_weights = 1 / (np.sum(m_uav_rates, axis=0) +
                                 self.epsilon_reweighting)

        #log.debug("m_uav_rates=", m_uav_rates)
        # print(m_uav_rates)
        return m_uav_rates, status

    def _min_uav_num_placement_scipy_reweight(self, m_capacity):

        m_uav_rates, status = self._reweighting_loop(
            m_capacity,
            lambda group_weights, prev_state: self._group_sparse_scipy(
                m_capacity, group_weights, prev_state))

        return m_uav_rates, status

    def _reweighting_loop(self, m_capacity, f_optimizer, b_debug=False):
        """
        
        Input:

            `f_optimizer` is a function of the form

                m_uav_rates, status, new_state = f_optimizer(group_weights,
                prev_state)

            where `prev_state` and `new_state` can be used arbitrarily by
            `f_optimizer`. When `f_optimizer` is invoked for the first time,
            `prev_state` is None.
        """

        # Reweighting [Candes, Wakin, and Boyd, 2008]
        # group_weights = 1 / (1 + np.sum(m_capacity >= self.min_user_rate) +
        #                      self.reweighting_noise * np.random.random(
        #                          (m_capacity.shape[1], )))
        group_weights = 1 / (1 +
                             np.sum(m_capacity >= self.min_user_rate, axis=0) +
                             self.reweighting_noise * np.random.random(
                                 (m_capacity.shape[1], )))
        group_weights = group_weights / np.sum(group_weights)

        if b_debug:
            print("Normalized group_weights: {}".format(group_weights /
                                                        np.max(group_weights)))

        if self.b_save_group_weights:
            np.save('output/tempt/group_weights.npy', group_weights)
        if self.b_load_group_weights:
            group_weights = np.load('output/tempt/group_weights.npy')

        state = None

        min_num_uav = np.ceil(
            np.round(
                m_capacity.shape[0] * self.min_user_rate /
                self.max_uav_total_rate, 2))

        # The three following variables are to handle the case when reaching the max number of reweighting_num_iter but the status of GroupSparseUAVPlacer is still "out of max iterations".
        l_num_uavs = []
        l_m_uav_rates = []

        for ind in range(self.reweighting_num_iter):

            m_uav_rates, status, state = f_optimizer(group_weights, state)

            # update
            if np.min(m_uav_rates) < 0:
                raise ValueError(
                    '`m_uav_rates` cannot contain negative entries.')

            group_weights = 1 / (np.max(m_uav_rates, axis=0) +
                                 self.epsilon_reweighting)

            group_weights = group_weights / np.sum(group_weights)

            num_uavs = state["num_uavs"]
            print(
                f"--- GroupSparseUAVPlacer: {ind} --- {status} --- {num_uavs}")

            if (self.backend == "admm") and (status == 'success'):
                l_num_uavs.append(num_uavs)
                l_m_uav_rates.append(m_uav_rates)

            if (state["num_uavs"] <= min_num_uav + 1) and (status
                                                           == 'success'):
                break

        if self.backend == "admm":
            if l_num_uavs == []:
                raise ValueError('Unsuccess placement')

            status = 'success'
            min_pos = np.argmin(l_num_uavs)
            m_uav_rates = l_m_uav_rates[min_pos]

        # print optimal values and infeasibility for m_uav_rates
        objective, total_infeasibility = self._check_feasibility(
            group_weights, m_capacity, m_uav_rates)
        print("Objective: {}, total infeasibility: {}".format(
            objective, total_infeasibility))
        v_inf_norms = np.linalg.norm(m_uav_rates, axis=0, ord=np.inf)
        print("Inf norms: ", v_inf_norms)

        return m_uav_rates, status

    def _group_sparse_scipy(self, m_capacity, group_weights, prev_state):

        # --------- this section has been looping ---------
        num_users, num_pts = m_capacity.shape

        A = -np.tile(np.eye(num_users), reps=(1, num_pts))
        #b = -np.tile([[self.min_user_rate]], (num_users,1 ))
        b = -np.tile([self.min_user_rate], (num_users, ))
        if self.max_uav_total_rate is not None:
            A = np.concatenate((
                A,
                np.repeat(np.eye(num_pts), num_users, axis=1),
            ),
                               axis=0)
            b = np.concatenate([
                b[:, None],
                np.tile([[self.max_uav_total_rate]], (num_pts, 1))
            ],
                               axis=0)[:, 0]
        U = m_capacity
        # ---------

        m_uav_rates, status = weighted_group_sparse_scipy(
            group_weights, A, b, U)

        return m_uav_rates, status, None

    def _min_uav_num_placement_admm(self, m_capacity):

        m_uav_rates, status = self._reweighting_loop(
            m_capacity, lambda group_weights, prev_state: self.
            _group_sparse_admm(m_capacity,
                               group_weights,
                               prev_state,
                               initial_error_tol=self.admm_initial_error_tol))

        return m_uav_rates, status

    def _group_sparse_admm(self,
                           m_capacity,
                           group_weights,
                           d_state,
                           initial_error_tol=10):

        b_debug = False

        def initialize_ZU(m_capacity, group_weights):
            mode = 1
            if mode == 1:
                m_U_current = np.ones(m_capacity.shape)
            elif mode == 2:
                m_U_current = -np.tile(group_weights.T / num_users / step_size,
                                       (num_users, 1))

            mode = 1
            if mode == 1:
                m_Z_previous = m_capacity * (m_capacity >= self.min_user_rate)

                # m_Z_previous = np.zeros(m_capacity.shape)
                # v_pos = np.array(list(set(list(np.argmax(m_capacity,
                #                                          axis=1)))))
                # m_Z_previous[:, v_pos] = m_capacity[:, v_pos]

            elif mode == 2:

                # TODO: modify to ensure feasibility

                # compute max_num_uavs
                # max_num_uavs = np.ceil(
                #     min_user_rate / max_uav_total_rate) * num_users

                # option 1: find grid points that provide the greatest capacity to the users
                v_pos = np.array(list(set(list(np.argmax(m_capacity,
                                                         axis=1)))))

                # option 2: find max_num_uavs grid points that are able to guarantee min_user_rate to as many users as possible.
                # v_pos = np.flip(
                #     np.argsort(np.sum(m_capacity >= self.min_user_rate,
                #                       axis=0)))[:int(max_num_uavs)]

                # find rate allocation
                m_Z_previous = self.find_rate_allocation(m_capacity,
                                                         min_user_rate,
                                                         max_uav_total_rate,
                                                         v_real_uav_inds=v_pos)
                if m_Z_previous is None:
                    raise ValueError("The initialization is infeasible")
            elif mode == 3:

                m_Z_previous = np.load('output/tempt/m_Z_previous.npy')
                m_U_current = np.load('output/tempt/m_U_current.npy')

            else:
                # use the Hammouti
                # pl_sr = SpaceRateKMeans(min_user_rate=min_user_rate,
                #                         max_uav_total_rate=max_uav_total_rate)
                pass

            b_save_admm_ZU = False
            # load from saved rate_allocation
            if b_save_admm_ZU:
                np.save('output/tempt/m_Z_previous.npy', m_Z_previous)
                np.save('output/tempt/m_U_current.npy', m_U_current)

            return m_Z_previous, m_U_current

        # initialize
        if d_state is None:
            m_Z_previous, m_U_current = initialize_ZU(m_capacity,
                                                      group_weights)
            d_state = dict()
            reweighting_iter = 0
            step_size = self.admm_stepsize
        else:
            m_Z_previous = d_state["m_Z"]
            m_U_current = d_state["m_U"]
            reweighting_iter = d_state["reweight_iter"]

        # step_size = self.admm_stepsize / np.power(
        #     2, np.floor((reweighting_iter % 25) / 5))
        step_size = self.admm_stepsize / np.power(
            2, np.floor((reweighting_iter % 25) / 5))

        if step_size < 1e-8:
            step_size = 1e-8

        # if (reweighting_iter > 0) and (reweighting_iter % 25 == 0):
        #     step_size = 1e-3 * (10 * np.sin(np.random.rand()))

        max_uav_total_rate = self.max_uav_total_rate
        min_user_rate = self.min_user_rate
        # step_size = self.admm_stepsize

        eps_abs = self.eps_abs
        eps_rel = self.eps_rel
        max_num_iter = self.admm_max_num_iter
        num_users, num_pts = m_capacity.shape
        v_rate_in_xstep = group_weights / (num_users * step_size)
        addition_bound = 0.001 * np.max(m_capacity)

        # the following parameters are used for checking the stopping criterion
        m_A_first = np.concatenate((np.eye(num_users), np.zeros(
            (num_users, 1))),
                                   axis=1)
        m_A_second = np.eye(num_pts)
        m_B_first = -np.eye(num_users)
        m_B_second = np.eye(num_pts)
        c_scale = np.sqrt(m_A_first.shape[0] * m_A_second.shape[1])

        def admm_single_iter(m_capacity,
                             group_weights,
                             m_Z_previous,
                             m_U_current,
                             error_tolerance=1e-6):

            def xstep(m_capacity, v_wg, m_Z, m_U):

                # checked
                def xstep_no_constraint(wg, v_sub):

                    if (wg < 0) or (step_size < 0):
                        raise ValueError(
                            'xstep_max_no_constraint has no solution')

                    def xstep_max_no_constraint(sg):
                        return np.sum(np.maximum(
                            (v_sub - sg), 0)) - wg / step_size

                    c_rate = wg / (num_users * step_size)
                    sg_min = np.min(v_sub) - c_rate - 0.1 * np.max(m_capacity)
                    sg_max = np.max(v_sub) - c_rate + 0.1 * np.max(m_capacity)

                    sg_next = bisect(xstep_max_no_constraint, sg_min, sg_max)

                    rg_next = np.minimum(v_sub, sg_next)

                    return rg_next

                # checked
                def xstep_constraint(wg, v_sub):

                    mu = (-step_size * max_uav_total_rate +
                          step_size * np.sum(v_sub) - wg) / num_users

                    # with max rate constraint
                    def xstep_max_constraint(sg):
                        return np.sum(
                            np.maximum(mu, step_size *
                                       (v_sub - sg))) - wg - mu * num_users

                    v_ones = np.ones((num_users, 1))

                    c_rate = wg / (num_users * step_size) + mu / step_size
                    sg_min = np.min(v_sub) - c_rate - 0.1 * np.max(m_capacity)
                    sg_max = np.max(v_sub) - c_rate + 0.1 * np.max(m_capacity)

                    sg_next = bisect(xstep_max_constraint, sg_min, sg_max)

                    return np.minimum(v_sub - (mu / step_size), sg_next)

                m_R = 0.01 * m_capacity

                # x step
                for index_g in range(num_pts):

                    v_sub = m_Z[:, index_g] - m_U[:, index_g]

                    # x step non constraint
                    m_R[:, index_g] = xstep_no_constraint(v_wg[index_g], v_sub)

                    if np.sum(m_R[:, index_g]) > max_uav_total_rate:

                        # x step with constraint
                        m_R[:,
                            index_g] = xstep_constraint(v_wg[index_g], v_sub)

                return m_R

            def f_bisec_vector(target_func,
                               v_lower_bounds,
                               v_upper_bounds,
                               max_num_iter=100):

                v_out_lower = target_func(v_lower_bounds) > 0
                v_out_upper = target_func(v_upper_bounds) > 0

                if not np.all(np.logical_xor(v_out_lower, v_out_upper)):
                    raise ValueError(
                        "Input to the bisection method is not valid.")

                for _ in range(1, max_num_iter + 1):

                    # bisection
                    v_avg = (v_upper_bounds + v_lower_bounds) / 2.0
                    v_out_avg = target_func(v_avg) > 0

                    # error_current describes the distance between our current solution v_avg and the desired solution.
                    # error_current = np.sum(abs(v_out_avg))
                    error_current = np.mean(v_upper_bounds - v_lower_bounds)

                    # early stop
                    if (error_current < error_tolerance):
                        return v_avg

                    # update v_lower_bounds
                    v_update_lower_bound_pos = np.logical_not(
                        np.logical_xor(v_out_avg, v_out_lower))
                    v_lower_bounds[v_update_lower_bound_pos] = v_avg[
                        v_update_lower_bound_pos]

                    # update v_upper_bounds
                    v_update_upper_bound_pos = np.logical_not(
                        np.logical_xor(v_out_avg, v_out_upper))
                    v_upper_bounds[v_update_upper_bound_pos] = v_avg[
                        v_update_upper_bound_pos]

                    v_out_lower = target_func(v_lower_bounds) > 0
                    v_out_upper = target_func(v_upper_bounds) > 0

                return v_avg

            def xstep_parallel(m_capacity, v_wg, m_Z, m_U):

                def xstep_no_constraint_parallel():

                    m_sub = m_Z - m_U

                    last_term = -v_wg / step_size + np.sum(m_sub, axis=0)

                    def xstep_max_no_constraint_parallel(v_s):
                        return np_sum(np_maximum(-v_s, -m_sub),
                                      axis=0) + last_term

                    # def xstep_max_no_constraint_parallel(v_s):
                    #     return np.sum(np.maximum(m_sub - v_s, 0),
                    #                   axis=0) - v_wg / step_size

                    # find v_s_min and v_s_max
                    v_s_min = np.min(m_sub,
                                     axis=0) - v_rate_in_xstep - addition_bound

                    v_s_max = np.max(m_sub,
                                     axis=0) - v_rate_in_xstep + addition_bound

                    v_s_next = f_bisec_vector(xstep_max_no_constraint_parallel,
                                              v_s_min, v_s_max)

                    ###### DEBUG ##########

                    # import timeit
                    # print(
                    #     "told = ",
                    #     timeit.timeit(lambda: f_bisec_vector(
                    #         xstep_max_no_constraint_parallel, v_s_min, v_s_max
                    #     ),
                    #                   number=10000))
                    # print(
                    #     "tnew = ",
                    #     timeit.timeit(lambda: f_bisec_vector(
                    #         xstep_max_no_constraint_parallel, v_s_min, v_s_max
                    #     ),
                    #                   number=10000))
                    # err = np.linalg.norm(
                    #     f_bisec_vector(xstep_max_no_constraint_parallel,
                    #                    v_s_min, v_s_max) -
                    #     f_bisec_vector_new(xstep_max_no_constraint_parallel,
                    #                        v_s_min, v_s_max))

                    # print(f"err = {err}")

                    return np.minimum(m_sub, v_s_next)

                def xstep_constraint_parallel(m_sub, v_mu, v_wg_subvec):

                    def xstep_max_constraint_parallel(v_s):
                        return np.sum(np.maximum(
                            v_mu,
                            step_size * (m_sub - np.expand_dims(v_s, axis=0))),
                                      axis=0) - v_wg_subvec - v_mu * num_users

                    v_sub = v_wg_subvec / (step_size *
                                           num_users) + v_mu / step_size

                    v_s_min = np.min(m_sub, axis=0) - v_sub - addition_bound

                    v_s_max = np.max(m_sub, axis=0) - v_sub + addition_bound

                    v_s_next = f_bisec_vector(xstep_max_constraint_parallel,
                                              v_s_min, v_s_max)

                    return np.minimum(
                        m_sub - np.expand_dims(v_mu, axis=0) / step_size,
                        np.expand_dims(v_s_next, axis=0))

                m_R = xstep_no_constraint_parallel()

                # check if the max_uav_total_rate is satisfied, xstep_constraint
                v_greater_max_rate_ind = np.where(
                    np.sum(m_R, axis=0) > max_uav_total_rate)[0]

                # v_wg = np.expand_dims(v_wg, axis=1)

                if len(v_greater_max_rate_ind) != 0:

                    m_Z_submat = m_Z[:, v_greater_max_rate_ind]
                    m_U_submat = m_U[:, v_greater_max_rate_ind]
                    v_wg_subvec = v_wg[v_greater_max_rate_ind]

                    m_sub = m_Z_submat - m_U_submat

                    v_mu = (-step_size * max_uav_total_rate + step_size *
                            np.sum(m_sub, axis=0) - v_wg_subvec) / num_users

                    m_R[:, v_greater_max_rate_ind] = xstep_constraint_parallel(
                        m_sub, v_mu, v_wg_subvec)
                    # m_R_submat = xstep_constraint_parallel(
                    #     m_sub, v_mu, v_wg_subvec)

                return m_R

            def zstep(m_capacity, m_R, m_U):

                c_rate = min_user_rate / m_capacity.shape[1]

                # checked
                def zstep_single(v_rm, v_um, v_cm):

                    v_sum = v_rm + v_um

                    def zstep_common_max(lamb):
                        return np.maximum(0, np.minimum(v_cm, v_sum - lamb))

                    def zstep_max(lamb):
                        return np.sum(zstep_common_max(lamb)) - min_user_rate

                    lamb_min = np.min(v_sum - v_cm) - 0.1 * np.max(m_capacity)

                    v_pos = np.where(v_cm > c_rate)[0]
                    lamb_max = np.max(
                        v_sum[v_pos]) - c_rate + 0.1 * np.max(m_capacity)

                    lamb = bisect(zstep_max, lamb_min, lamb_max)

                    return zstep_common_max(lamb)

                m_Z = np.zeros(m_capacity.shape)

                # z step
                for index_m in range(num_users):

                    # find z
                    m_Z[index_m] = zstep_single(m_R[index_m], m_U[index_m],
                                                m_capacity[index_m])

                return m_Z

            def zstep_parallel(m_capacity, m_R, m_U):

                m_sum = m_R + m_U
                m_tempt = m_capacity - m_sum
                v_tempt = np.sum(m_sum, axis=1) - min_user_rate

                def f_zstep_common_max(v_lamb):
                    return np_maximum(
                        -m_sum,
                        np_minimum(m_tempt, -np_expand_dims(v_lamb, axis=1)))

                # def f_zstep_common_max(v_lamb):
                #     return np_maximum(
                #         0,
                #         np_minimum(m_capacity,
                #                    m_sum - np_expand_dims(v_lamb, axis=1)))

                def zstep_max_parallel(v_lamb):

                    return np.sum(f_zstep_common_max(v_lamb), axis=1) + v_tempt

                # old version
                v_lamb_min = np.min(m_sum - m_capacity, axis=1)

                c_rate = min_user_rate / num_pts

                ll_pos = [np.where(v_cm > c_rate)[0] for v_cm in m_capacity]

                v_lamb_max = np.array([
                    np.max(m_sum[ind, ll_pos[ind]])
                    for ind in range(m_capacity.shape[0])
                ])
                # end of old version

                v_lamb = f_bisec_vector(zstep_max_parallel, v_lamb_min,
                                        v_lamb_max)

                return np_maximum(
                    -m_sum, np_minimum(
                        m_tempt, -np_expand_dims(v_lamb, axis=1))) + m_sum

            # m_R = xstep(m_capacity, group_weights, m_Z_previous, m_U_current)

            m_R = xstep_parallel(m_capacity, group_weights, m_Z_previous,
                                 m_U_current)

            # m_Z_current = zstep(m_capacity, m_R, m_U_current)

            m_Z_current = zstep_parallel(m_capacity, m_R, m_U_current)

            # u step
            m_U_current = m_U_current + m_R - m_Z_current

            return m_R, m_Z_current, m_U_current

        def stopping_criterion_satisfied(m_R,
                                         m_Z_current,
                                         m_Z_previous,
                                         m_Y,
                                         eps_abs,
                                         eps_rel,
                                         debug=False):

            v_s = np.linalg.norm(m_R, ord=inf, axis=0)
            m_X = np.concatenate((m_R.T, np.expand_dims(v_s, axis=1)),
                                 axis=1).T

            m_tempt_1 = m_A_first @ m_X @ m_A_second
            m_tempt_2 = m_B_first @ m_Z_current @ m_B_second

            eps_pri = c_scale * eps_abs + eps_rel * np.max(
                (np.linalg.norm(m_tempt_1), np.linalg.norm(m_tempt_2)))

            # eps_pri = np.sqrt(
            #     m_A_first.shape[0] *
            #     m_A_second.shape[1]) * eps_abs + eps_rel * np.max(
            #         (np.linalg.norm(m_A_first @ m_X @ m_A_second),
            #          np.linalg.norm(m_B_first @ m_Z_current @ m_B_second)))

            eps_dual = c_scale * eps_abs + eps_rel * np.linalg.norm(
                m_A_first.T @ m_Y @ m_A_second.T)
            # eps_dual = np.sqrt(m_A_first.shape[1] * m_A_second.shape[0]
            #                    ) * eps_abs + eps_rel * np.linalg.norm(
            #                        m_A_first.T @ m_Y @ m_A_second.T)

            # calculate the primal residual
            m_Q = m_tempt_1 + m_tempt_2
            # m_Q = m_A_first @ m_X @ m_A_second + m_B_first @ m_Z_current @ m_B_second

            # calculate the dual residual
            # m_P = step_size * m_A_first.T @ (
            #     m_tempt_2 -
            #     m_B_first @ m_Z_previous @ m_B_second) @ m_A_second.T
            m_P = step_size * m_A_first.T @ m_B_first @ (
                m_Z_current - m_Z_previous) @ m_B_second @ m_A_second.T

            if debug:
                print("--- Primal norm: {}, --- epsilon: {}".format(
                    np.linalg.norm(m_Q), eps_pri))
                print("--- Dual norm: {}, --- epsilon: {}".format(
                    np.linalg.norm(m_P), eps_dual))

            if (np.linalg.norm(m_Q) <= eps_pri) and (np.linalg.norm(m_P) <=
                                                     eps_dual):
                # is_epsilons_stop = True
                return True
            else:
                # is_epsilons_stop = False
                return False

            # return is_epsilons_stop

        def plot_process(v_objective_value, v_total_infea):
            x_axis = np.linspace(1,
                                 len(v_objective_value) - 1,
                                 len(v_objective_value) - 1)
            fig, axs = plt.subplots(2)
            # axs[0].plot(x_axis, v_objective_value[1:], label='objective')
            axs[0].plot(x_axis, v_objective_value[1:], label='objective')
            axs[0].set_yscale('log')
            axs[0].set_title('Objective value')
            axs[0].set_xlabel("Iteration")
            axs[0].grid()

            axs[1].plot(x_axis, v_total_infea[1:], label='infeasibility')
            axs[1].set_yscale('log')
            axs[1].set_title('Infeasibility value')
            axs[1].set_xlabel("Iteration")
            axs[1].grid()

            plt.show()

            print("Plotted")

        v_objective_value = [0]
        v_total_infea = [0]

        # prof = cProfile.Profile()
        # prof.enable()

        # if reweighting_iter < 2:
        #     error_tolerance = 1e3 / np.power(10, reweighting_iter)
        # else:
        #     error_tolerance = 1e2

        for ind_iter in range(max_num_iter):

            if self.b_admm_decrease_err_tol:
                error_tolerance = initial_error_tol / (ind_iter + 1)
            else:
                error_tolerance = 1

            m_rate, m_Z_current, m_U_current = admm_single_iter(
                m_capacity,
                group_weights,
                m_Z_previous,
                m_U_current,
                error_tolerance=error_tolerance)

            # check results
            if b_debug:
                objective, total_infeasibility = self._check_feasibility(
                    group_weights, m_capacity, m_rate)
                v_objective_value.append(objective)
                v_total_infea.append(total_infeasibility)
                num_uavs = np.sum(
                    np.sum(group_sparsify(m_rate, self.sparsity_tol), axis=0) >
                    0)

                if np.mod(ind_iter, 50) == 0:
                    print(
                        f"Iteration {ind_iter}, num_uavs: {num_uavs}, objective: {objective}, infeasibility: {total_infeasibility}"
                    )

            if stopping_criterion_satisfied(m_rate, m_Z_current, m_Z_previous,
                                            step_size * m_U_current, eps_abs,
                                            eps_rel):
                status = 'success'
                if b_debug:
                    print("Status: " + status)

                    print(
                        "Terminate at iteration {}, objective: {}, infeasibility: {}"
                        .format(ind_iter, objective, total_infeasibility))

                    print("Num uavs: {}".format(num_uavs))

                break

            # update Z
            m_Z_previous = m_Z_current

        # prof.disable()
        # prof.dump_stats('output/profiled_admm.prof')

        # end of profiling
        objective, total_infeasibility = self._check_feasibility(
            group_weights, m_capacity, m_rate)

        if ind_iter == max_num_iter - 1:
            if total_infeasibility < self.admm_feasibility_tol:
                status = 'success'
            else:
                status = 'out of max iterations'

        num_uavs = np.sum(
            np.sum(group_sparsify(m_rate, self.sparsity_tol), axis=0) > 0)

        if self.b_plot_progress:
            plot_process(v_objective_value, v_total_infea)

        d_state["m_Z"] = m_Z_current
        d_state["m_U"] = m_U_current
        d_state["reweight_iter"] = reweighting_iter + 1
        d_state["num_uavs"] = num_uavs
        d_state["tot_infeasibility"] = total_infeasibility

        # Get rid of potential negative entries
        m_rate[m_rate < 0] = 0

        return m_rate, status, d_state

    def _check_feasibility(self,
                           group_weights,
                           m_capacity,
                           m_uav_rates,
                           debug=False):

        v_rmax = self.max_uav_total_rate * np.ones((m_capacity.shape[1], 1))
        v_s = np.linalg.norm(m_uav_rates, axis=0, ord=np.inf)

        # objective
        objective = group_weights.T @ v_s

        # check max uav total rate
        check_r_max = np.linalg.norm(
            np.maximum(
                m_uav_rates.T @ np.ones((m_uav_rates.shape[0], 1)) - v_rmax,
                np.zeros((v_rmax.size, 1))), 2)

        # check min user's rate
        check_r_min = np.linalg.norm(
            m_uav_rates @ np.ones(
                (m_uav_rates.shape[1], 1)) - self.min_user_rate, 2)

        # check capacity bound
        check_lower_bound = np.linalg.norm(
            np.matrix.flatten(
                np.maximum(-m_uav_rates, np.zeros(m_uav_rates.shape))), 2)
        check_upper_bound = np.linalg.norm(
            np.matrix.flatten(
                np.maximum(m_uav_rates - m_capacity,
                           np.zeros(m_uav_rates.shape))), 2)
        check_bound_rate = check_lower_bound + check_upper_bound

        # check infinity rate
        check_s = np.linalg.norm(
            np.matrix.flatten(
                np.maximum(
                    m_uav_rates - np.ones(
                        (m_uav_rates.shape[0], 1)) @ np.expand_dims(v_s,
                                                                    axis=1).T,
                    np.zeros(m_uav_rates.shape))), 2)
        if debug:
            print("Optimal value: ", objective)
            print("--------- r_max: ", check_r_max)
            print("--------- r_min: ", check_r_min)
            print("--------- cap. bound: ", check_bound_rate)
            print("--------- inf. norm: ", check_s)

        total_infeasibility = check_r_max + check_r_min + check_bound_rate + check_s

        return objective, total_infeasibility


class FromFixedNumUavsPlacer(CapacityBasedPlacer):
    """ 
    This abstract class contains CapacityBasedPlacers whose placement primitive
    requires setting a number of UAVs. 
    
    If self.num_uavs is not provided, then the number of uavs is gradually
    increased until the minimum user rate exceeds `self.min_user_rate` or the
    number of uavs reaches self.max_num_uavs.

    """

    # If `_place_num_uavs` fails, the following functions are tried. It must be a list of functions.
    _last_resort_place_num_uavs = None

    def __init__(self, max_num_uavs=30, num_uavs=None, **kwargs):
        """ Args:
            
                `max_num_uavs`: when self.min_user_rate is not None, the number of UAVs attempted is 
                gradually increased until min(num_users, max_num_uavs)

        """
        super().__init__(**kwargs)

        self.max_num_uavs = max_num_uavs
        self.num_uavs = num_uavs
        assert xor(self.min_user_rate is None, self.num_uavs is None)

    def place(self, fly_grid, channel, user_coords, *args, **kwargs):
        map = channel.capacity_map(
            grid=fly_grid,
            user_coords=user_coords,
        )
        if self.min_user_rate is not None:
            ind_uavs = self._place_by_increasing_num_uavs(
                *args, map=map, user_coords=user_coords, **kwargs)
        else:
            ind_uavs = self._place_num_uavs(map=map,
                                            user_coords=user_coords,
                                            num_uavs=self.num_uavs)
        if ind_uavs is None:
            return None
        return map.grid.list_pts()[ind_uavs]

    def _place_by_increasing_num_uavs(self, *args, **kwargs):
        """Runs self._place_num_uavs with an increasing num_uavs until all users
        receive the minimum rate. The latter condition holds when each user
        receives `self.min_user_rate` from all UAVs combined. If you want this
        condition to be satisfied when each user connects only to the strongest
        UAV, then set `channel.min_link_capacity=self.min_user_rate`."""

        inds_uavs = self._place_by_increasing_num_uavs_from_fun(
            self._place_num_uavs, *args, **kwargs)

        if inds_uavs is None:
            log.warning(
                f"Impossible to guarantee minimum rate in {self.__class__.__name__}"
            )

        if self._last_resort_place_num_uavs is not None:
            for fun in self._last_resort_place_num_uavs:
                log.warning("   Using last resort procedure")
                inds_uavs = self._place_by_increasing_num_uavs_from_fun(
                    fun, *args, **kwargs)

                if inds_uavs is None:
                    log.warning(
                        f"    Last resort procedure also failed in {self.__class__.__name__}"
                    )
        return inds_uavs

    # def is_m_rate_feasible(self, m_rate_submat):
    #     """ `m_rate_submat`: num_users x num_uavs where the (i,j)-th entry is
    #     the rate that the j-th uav allocates to the i-th user.

    #     It returns True iff the user and uav rate constraints are met. It is not
    #     checked that the entries are between 0 and the capacity.
    #     """

    #     b_all_users_get_min_rate = all(
    #         np.sum(m_rate_submat, axis=1) >= self.min_user_rate)
    #     b_all_uavs_satisfy_rate_constraint = (
    #         self.max_uav_total_rate is None) or all(
    #             np.sum(m_rate_submat, axis=0) <= self.max_uav_total_rate)

    #     return b_all_users_get_min_rate and b_all_uavs_satisfy_rate_constraint

    def _place_by_increasing_num_uavs_from_fun(self,
                                               fun,
                                               map,
                                               user_coords=None):

        if self.max_uav_total_rate is not None:

            #assert self.max_uav_total_rate >= self.min_user_rate, "Cannot guarantee a min user rate"

            max_num_uavs = int(
                np.ceil(self.min_user_rate / self.max_uav_total_rate) *
                len(user_coords))

            min_num_uavs = int(
                np.ceil(
                    len(user_coords) * self.min_user_rate /
                    self.max_uav_total_rate))
        else:
            max_num_uavs = len(user_coords)
            min_num_uavs = 1

        for num_uavs in range(min_num_uavs, max_num_uavs + 1):

            inds = fun(map=map, user_coords=user_coords, num_uavs=num_uavs)

            # m_rate = self.find_rate_allocation(
            #     m_capacity[:, inds],
            #     min_user_rate=self.min_user_rate,
            #     max_uav_total_rate=self.max_uav_total_rate)

            if (inds is not None):
                # num_served_user, _ = self._check_association(m_capacity[:,
                #                                                         inds])

                return inds

        #raise ValueError("Infeasible")
        # return None
        return map.grid.nearest_inds(user_coords)

    # Abstract methods
    def _place_num_uavs(self, map, user_coords, num_uavs):
        """ Returns a vector of indices of the grid points where UAVs need to be
        placed. If no feasible placement exists, then it returns None. 
        """
        raise NotImplementedError

    def _check_association(self, m_capacity_submat, mode=None, debug=False):

        num_users, num_uavs = m_capacity_submat.shape

        # compute the number of served users, for a given uav placement
        """            
        This function associates each user to the UAV from which it receives the
        greatest rate. The association takes into account min_user_rate and
        max_links_per_uav.

        Given:  m_capacity:            (num_users x num_pts)
                v_pos_uav_in_grid:     (1 x num_uavs) a vector of grid points where
                                    the UAVs are placed
                self.min_user_rate
                self.max_uav_total_rate
        
        Return: num_served_users

        We need not consider the case where a user is de-associated by 
        a uav but possible to associate with another uav because of some 
        reasons.
            1. This situation might be covered in another solution in the 
            current generation.
            2. For each UAV, simultaneously adding one and de-associating 
            one user does not change the total number of served users.
            3. [shehzad2021backhaul] also does not consider that case.
        """

        assert self.min_user_rate != 0
        max_links_per_uav = np.floor(self.max_uav_total_rate /
                                     self.min_user_rate)

        # initialize association matrix
        m_associate = np.zeros((num_users, num_uavs))
        v_num_ser_usr_per_uav = np.sum(m_associate, 0)
        # each user connects to only 1 UAV
        for ind_user in range(num_users):

            if not all(m_capacity_submat[ind_user, :] < self.min_user_rate):

                if mode == "max":
                    ind_pos = np.argmax(m_capacity_submat[ind_user, :])
                else:
                    v_pos = np.where(
                        m_capacity_submat[ind_user, :] >= self.min_user_rate)

                    ind_pos = v_pos[0][np.argmin(v_num_ser_usr_per_uav[v_pos])]

                m_associate[ind_user, ind_pos] = 1

                # update numbers of served users
                v_num_ser_usr_per_uav = np.sum(m_associate, 0)

        # de-associate
        for ind_uav in range(num_uavs):
            if v_num_ser_usr_per_uav[ind_uav] > max_links_per_uav:
                num_red_usrs = v_num_ser_usr_per_uav[
                    ind_uav] - max_links_per_uav

                m_associate[
                    np.argsort(m_associate[:, ind_uav])[-int(num_red_usrs):],
                    ind_uav] = 0

        v_num_ser_usr_per_uav = np.sum(m_associate, 0)

        num_ser_usr = np.sum(m_associate)

        v_redundant_uavs = None

        # remove redundant uavs if there are any
        if not all(v_num_ser_usr_per_uav != 0):

            if debug:
                print("There are some redundant uavs")

            # v_pos_uav_in_grid = np.delete(v_pos_uav_in_grid,
            #                               v_num_ser_usr_per_uav == 0)
            v_redundant_uavs = np.where(v_num_ser_usr_per_uav == 0)[0]

            m_associate = np.delete(m_associate, v_redundant_uavs, axis=1)
            v_num_ser_usr_per_uav = np.delete(v_num_ser_usr_per_uav,
                                              v_redundant_uavs)

        if debug:
            print("Number of served users: {}".format(
                np.sum(v_num_ser_usr_per_uav)))
            print("m_associate: \n", m_associate)

        return int(num_ser_usr), v_redundant_uavs

    def _associate_users_gale_shapley_like(self,
                                           m_capacity_submat,
                                           debug=False):
        """ 
        Given the capacities between the users and the uavs, this function
        associates users with uavs by establishing preferences among them in
        the decreasing order of capacity. The function returns a vector that
        has num_users entries. If the i-th entry is k, it means that the
        i-th user must associate with the uav that corresponds to the k-th
        column of m_capacity.
        
        Given:  m_capacity_submat:     (num_users x num_uavs)
        
        Returns: 
        
            `v_ass_inds_rel_to_v_uav_inds`: 
            
                - If there is a feasible association, this is a vector of
                    length
                num_users where the n-th entry is the index of the UAV with
                which user n must associate. The UAV index is relative to the
                column of m_capacity_submat, i.e., the m-th UAV corresponds to
                the m-th column of m_capacity_submat.

                - If there is no feasible association, this is None.

            Feasible association means that all users receive at least
            self.min_user_rate and the total rate that each UAV provides to
            all its associated users does not exceed
            self.max_uav_total_rate.

        """

        max_uav_users = np.floor(self.max_uav_total_rate / self.min_user_rate)

        num_users, num_uavs = m_capacity_submat.shape

        def vcapacity_to_preferences(v_capacity):
            #
            v_inds = np.flip(np.argsort(v_capacity))
            return [
                ind for ind in v_inds if v_capacity[ind] >= self.min_user_rate
            ]

        def has_empty_lists(ll):
            return np.any([len(l) == 0 for l in ll])

        # Obtain user and uav preference lists in the decreasing order,
        # take into account min_user_rate
        ll_user_preferences = [
            vcapacity_to_preferences(v_capacity)
            for v_capacity in m_capacity_submat
        ]
        if has_empty_lists(ll_user_preferences):
            return None

        ll_uav_preferences = [
            vcapacity_to_preferences(v_capacity)
            for v_capacity in m_capacity_submat.T
        ]

        v_num_assigned_users = np.zeros(num_uavs)
        m_associate = np.zeros((num_users, num_uavs))

        l_unassigned_users = list(range(num_users))

        while len(l_unassigned_users) != 0:
            ind_user = l_unassigned_users.pop()
            l_preferred_uavs = ll_user_preferences[ind_user]

            for ind_uav in l_preferred_uavs:
                if v_num_assigned_users[ind_uav] < max_uav_users:
                    m_associate[ind_user, ind_uav] = 1
                    v_num_assigned_users[ind_uav] += 1
                    break
                else:

                    l_preferred_users = ll_uav_preferences[ind_uav]

                    rank_of_ind_user = np.where(
                        np.array(l_preferred_users) == ind_user)[0][0]

                    # find the users that are currently assigned to uav ind_uav
                    l_assigned_users = np.where(m_associate[:,
                                                            ind_uav] == 1)[0]

                    # find the ranks of the users assigned to uav ind_uav
                    l_ranks_of_assigned_users = [
                        np.where(
                            np.array(l_preferred_users) == ind_assigned_user)
                        [0][0] for ind_assigned_user in l_assigned_users
                    ]

                    # find the least preferred user assigned to uav ind_uav
                    least_preferred_assigned_user = l_assigned_users[np.argmax(
                        l_ranks_of_assigned_users)]

                    # find the rank of the least preferred assigned user in
                    # the list of preferred users of uav ind_uav
                    rank_of_least_preferred_assigned_user = np.max(
                        l_ranks_of_assigned_users)

                    if rank_of_least_preferred_assigned_user > rank_of_ind_user:
                        # unassign least_preferred_assigned_user from uav ind_uav
                        m_associate[least_preferred_assigned_user, ind_uav] = 0
                        # add user least_preferred_assigned_user to l_unassigned_users
                        l_unassigned_users.append(
                            least_preferred_assigned_user)
                        # assign user ind_user to uav ind_uav
                        m_associate[ind_user, ind_uav] = 1
                        break

            # return None

        l_ass_inds_rel_to_v_uav_inds = [
            np.where(v_associate == 1)[0] for v_associate in m_associate
        ]

        if has_empty_lists(l_ass_inds_rel_to_v_uav_inds):
            return None

        if debug:
            v_num_ser_usr_per_uav = np.sum(m_associate, axis=0)
            print(f"Number of served users: {np.sum(v_num_ser_usr_per_uav)}")
            print("m_associate: \n", m_associate)

        return np.array(l_ass_inds_rel_to_v_uav_inds)[:, 0]


class KMeansPlacer(FromFixedNumUavsPlacer):
    _name_on_figs = "Galkin et al."

    def __init__(self, max_uav_total_rate=None, **kwargs):
        """ Args:
    
        """
        super().__init__(**kwargs)
        self.max_uav_total_rate = max_uav_total_rate
        # self.max_links_per_uav = np.ceil(self.max_uav_total_rate/self.min_user_rate)

    # @staticmethod
    def _place_num_uavs(self, map, user_coords, num_uavs):

        print(f"--- KMeansPlacer --- num uavs: {num_uavs}")

        if self.max_uav_total_rate == None:
            assert user_coords.shape[1] == 3
            kmeans = KMeans(n_clusters=num_uavs)
            kmeans.fit(user_coords)
            centers = kmeans.cluster_centers_
            return map.grid.nearest_inds(centers)
        else:
            m_capacity = map.list_vals().T
            num_users, num_pts = m_capacity.shape

            assert user_coords.shape[1] == 3
            kmeans = KMeans(n_clusters=num_uavs)
            kmeans.fit(user_coords)

            v_centers = kmeans.cluster_centers_

            v_centers_pts = map.grid.nearest_inds(v_centers)

            # num_ser_users, _ = self._check_association(
            #     m_capacity[:, v_centers_pts])
            # if num_ser_users == num_users:
            #     return v_centers_pts
            # else:
            #     return None

            v_ass_inds_rel_to_v_uav_inds = self._associate_users_gale_shapley_like(
                m_capacity[:, v_centers_pts])

            if v_ass_inds_rel_to_v_uav_inds is not None:
                return v_centers_pts
            else:
                return None


class SpaceRateKMeans(FromFixedNumUavsPlacer):
    """ This is our best attempt to implement the modified K-means algorithm in
        [hammouti2019mechanism]. Unfortunately, some parts are not clear from
        the paper and the authors did not provide us with the code.

        This placer aims at maximizing the sum rate. The UAV positions are
        quantized to the fly grid to enforce no-fly zones.

        The placement assumes that each user connects to one and only one UAV.

        At each stage:

        1 - each user associates with a UAV. If `self.max_uav_total_rate` is
        None, then each user associates with the UAV from which it receives the
        greatest rate. Else, a Gale-Shapley-like algorithm is used. 

        2- each UAV is moved to the grid point that lies closest to the
        arithmetic mean of the coordinates of associated users.
    """
    _name_on_figs = "Hammouti et al."

    def __init__(self,
                 num_max_iter=20,
                 use_kmeans_as_last_resort=False,
                 **kwargs):
        """ Args:
    
        """
        super().__init__(**kwargs)
        self.num_max_iter = num_max_iter

        if use_kmeans_as_last_resort:
            self._last_resort_place_num_uavs = [KMeansPlacer._place_num_uavs]

    def _place_num_uavs(self, map, user_coords, num_uavs):

        def _associate_users_no_uav_rate_limit(m_capacity_submat):
            return np.argmax(m_capacity_submat, axis=1)

        def _associate_users(v_uav_inds):
            """Returns: 
            
                - a list of length num_users where the n-th entry indicates the
                index of the UAV to which user n must be associated. The entries
                of this list are therefore taken from v_uav_inds. 
                
                """

            m_capacity_submat = m_capacity[:, v_uav_inds]

            if self.max_uav_total_rate is None:
                v_ass_inds_rel_to_v_uav_inds = _associate_users_no_uav_rate_limit(
                    m_capacity_submat)
            else:
                v_ass_inds_rel_to_v_uav_inds = self._associate_users_gale_shapley_like(
                    m_capacity_submat)
                if v_ass_inds_rel_to_v_uav_inds is None:
                    v_ass_inds_rel_to_v_uav_inds = _associate_users_no_uav_rate_limit(
                        m_capacity_submat)
            return v_uav_inds[v_ass_inds_rel_to_v_uav_inds]

        def _place_uavs_at_centroids(v_ass_inds):
            """Returns:

                `m_uav_coords`: num_uavs x 3 matrix with the coordinates of the
                UAVs. The coordinates of each UAV are the arithmetic means of
                the coordinates of the UAVs associated with that UAV. The rows
                do not follow any specific order. 

            """

            m_uav_coords = np.array([
                np.mean(user_coords[np.where(v_ass_inds == ind_uav)[0], :],
                        axis=0) for ind_uav in set(v_ass_inds)
            ])

            return m_uav_coords

        m_capacity = map.list_vals().T

        num_users = len(user_coords)

        print(f"--- SpateRateKMeans --- num_uavs: {num_uavs}")

        # Initial random set of UAV coordinates
        v_rnd_user_inds = rng.choice(num_users, (num_uavs, ), replace=False)
        m_uav_coords = user_coords[v_rnd_user_inds, :]
        v_uav_inds = np.sort(map.grid.nearest_inds(m_uav_coords))

        for _ in range(self.num_max_iter):

            v_ass_inds = _associate_users(v_uav_inds)
            m_uav_coords = _place_uavs_at_centroids(v_ass_inds)
            v_uav_inds_new = np.sort(map.grid.nearest_inds(m_uav_coords))

            if np.all(v_uav_inds_new == v_uav_inds):
                #print(f"Finishing after {_+1} iterations")
                if self.max_uav_total_rate is None:
                    return v_uav_inds_new
                else:
                    v_ass_inds_rel_to_v_uav_inds = self._associate_users_gale_shapley_like(
                        m_capacity[:, v_uav_inds_new])
                    if v_ass_inds_rel_to_v_uav_inds is not None:
                        return v_uav_inds_new
                    else:
                        return None

            v_uav_inds = v_uav_inds_new

        log.warning("Maximum number of iterations reached at SpaceRateKMeans")

        # return v_uav_inds
        return None


class GridRatePlacer(FromFixedNumUavsPlacer):
    """ The placement assumes that each user connects to one and only one UAV.
    The goal is to maximize the sum rate. 

    At each stage, 

    1 - each user associates with the UAV from which it receives the greatest
    rate. 

    2- each UAV is moved to the grid point that maximizes the rate to its
    associated users. 

    In this way, the sum rate should never decrease, which implies that the
    algorithm  eventually converges.
    """

    def __init__(self, num_max_iter=200, num_initializations=20, **kwargs):
        """ Args:
    
        """
        super().__init__(**kwargs)
        self.num_max_iter = num_max_iter
        self.num_initializations = num_initializations
        assert self.max_uav_total_rate is None, "this placer cannot guarantee a max total rate per UAV"

    def _place_num_uavs(self, map, user_coords, num_uavs):
        m_capacity = map.list_vals().T

        def sum_rate(v_inds):
            # Each user connects to only 1 UAV.
            return np.sum(np.max(m_capacity[:, v_inds], axis=1))

        v_uav_inds = None
        for _ in range(self.num_initializations):
            v_uav_inds_new = self._place_num_uavs_one_initialization(
                m_capacity, user_coords, num_uavs)
            #print("new rate = ", sum_rate(v_uav_inds_new))
            if (v_uav_inds is None) or (sum_rate(v_uav_inds_new) >
                                        sum_rate(v_uav_inds)):
                v_uav_inds = v_uav_inds_new
        return v_uav_inds

    def _place_num_uavs_one_initialization(self,
                                           m_capacity,
                                           user_coords,
                                           num_uavs,
                                           debug=0):

        def _associate_users(v_uav_inds):
            """Returns a list of length num_users where the n-th entry indicates the
            index of the UAV to which user n must be associated. The entries of
            this list are therefore taken from v_uav_inds. """

            v_ass_inds_rel_to_v_uav_inds = np.argmax(m_capacity[:, v_uav_inds],
                                                     axis=1)
            v_ass_inds = v_uav_inds[v_ass_inds_rel_to_v_uav_inds]
            return v_ass_inds

        def _place_uavs_to_maximize_sum_rate_of_associated_users(v_ass_inds):
            """Returns:

                `v_gridpt_inds`: list of length `num_uavs` indicating the indices
                of the grid points where UAVs need to be placed. Each UAV is placed
                at the grid point that maximizes the sum rate of the associated users.
                """
            l_gridpt_inds = []
            for ind_uav in set(v_ass_inds):
                sum_rate_per_gridpt = np.sum(
                    m_capacity[np.where(v_ass_inds == ind_uav)[0], :], axis=0)
                new_gridpt = np.argmax(sum_rate_per_gridpt)
                l_gridpt_inds.append(new_gridpt)

            return np.array(l_gridpt_inds)

        num_users, num_gridpts = m_capacity.shape
        assert num_uavs <= num_users

        # Initial random set of UAV indices
        v_uav_inds = rng.choice(num_gridpts, (num_uavs, ), replace=False)
        sum_rate = None
        if debug:
            print("New initialization ----------")
        for _ in range(self.num_max_iter):
            v_ass_inds = _associate_users(v_uav_inds)
            v_uav_inds = _place_uavs_to_maximize_sum_rate_of_associated_users(
                v_ass_inds)

            sum_rate_new = np.sum(m_capacity[range(0, num_users), v_ass_inds])

            if (sum_rate is not None) and (sum_rate == sum_rate_new):
                #if np.all(v_uav_inds_new == v_uav_inds): # this check apparently leads to oscillations --> better check sum rate
                #print(f"Finishing after {_+1} iterations")
                return v_uav_inds
            sum_rate = sum_rate_new

            if debug:
                print("sum rate = ", sum_rate)

        raise ValueError("Maximum number of iterations reached")


class GeneticPlacer(FromFixedNumUavsPlacer):
    """ This placement assumes that each user connects to one and only one UAV.
    The goal is to maximize the sum rate.

    The objective (or fitness) function is the sum rate or, equivalently, the 
    number of users served by the UAVs as each user demands a min rate, which is 
    known.
    
    A genetic algorithm (GA) optimizes locations of the UAVs to maximize the sum
    rate in each iteration. 

    In this way, the sum rate (or the number of served users) should never 
    decrease, which implies that the algorithm  eventually converges.
    """

    def __init__(self,
                 max_num_gens=20,
                 mut_rate=0.05,
                 num_sols_in_pop=20,
                 percent_elite=0.4,
                 max_uav_total_rate=None,
                 **kwargs):
        """ Args:
    
        """
        super().__init__(**kwargs)
        self.max_num_gens = max_num_gens
        self.mut_rate = mut_rate
        self.num_sols_in_pop = num_sols_in_pop
        self.percent_elite = percent_elite
        self.max_uav_total_rate = max_uav_total_rate

    def _place_num_uavs(self, map, user_coords, num_uavs, debug=False):
        # Run num_gens generations. Each generation consists of num_sol solutions

        print(f"--- GeneticPlacer --- num_uavs: {num_uavs}")

        m_capacity = map.list_vals().T
        num_users, num_pts = m_capacity.shape

        max_uav_pos_in_grid = num_pts - 1
        num_rest = int(self.percent_elite * self.num_sols_in_pop)
        num_elite = self.num_sols_in_pop - num_rest

        def cal_num_ser_usr_mult_sols(m_pos_uavs_in_grid):

            # compute fitness values + associate users for the whole set of
            # solutions
            v_num_ser_usrs = np.zeros(self.num_sols_in_pop)
            for ind_sol in range(self.num_sols_in_pop):

                v_num_ser_usrs[ind_sol], _ = self._check_association(
                    m_capacity[:, m_pos_uavs_in_grid[ind_sol, :]])
                # v_num_ser_usrs[
                #     ind_sol], _ = self._associate_users_single_solution(
                #         m_capacity, m_pos_uavs_in_grid[ind_sol])

            return v_num_ser_usrs

        def gen_new_gen(m_pos_uavs_in_grid, v_num_ser_usrs):

            # choosing elite solutions
            m_elt_sols = np.zeros((num_elite, num_uavs)).astype(int)
            v_arg_sort = np.argsort(v_num_ser_usrs)
            for ind in range(num_elite):
                pos = int(v_arg_sort[self.num_sols_in_pop - ind - 1])
                m_elt_sols[ind] = m_pos_uavs_in_grid[pos]

            # choosing and mutate
            m_new_sols = np.zeros((num_rest, num_uavs)).astype(int)
            v_ind_sol = np.linspace(0, self.num_sols_in_pop - 1,
                                    self.num_sols_in_pop).astype(int)
            v_selec_prob = v_num_ser_usrs / np.sum(v_num_ser_usrs)
            v_choo_sols = np.random.choice(v_ind_sol, num_rest, p=v_selec_prob)
            for ind in range(num_rest):
                m_new_sols[ind] = m_pos_uavs_in_grid[v_choo_sols[ind]]
            # mutate
            m_mut_sols = m_new_sols + (np.random.randint(0, 1) * 2 -
                                       1) * np.random.randint(
                                           np.ceil(self.mut_rate * num_pts),
                                           size=(num_rest, num_uavs))
            # if m_mut_pos_uavs > num_pts, then m_mut_pos_uavs = max_pos
            m_mut_sols = m_mut_sols*(m_mut_sols<=max_uav_pos_in_grid) + \
                            max_uav_pos_in_grid*(m_mut_sols>max_uav_pos_in_grid)

            # if m_mut_pos_uavs < 0, then m_mut_pos_uavs = 0
            m_mut_sols = m_mut_sols * (m_mut_sols >= 0)

            m_new_gen = np.concatenate((m_elt_sols, m_mut_sols), 0)

            # return the new generation
            return m_new_gen.astype(int)

        # initialize a population
        m_pos_uavs_in_grid = np.zeros(
            (self.num_sols_in_pop, num_uavs)).astype(int)
        v_pos_samples = np.linspace(0, max_uav_pos_in_grid,
                                    num_pts).astype(int)
        for ind in range(self.num_sols_in_pop):
            # assert if the generated uav locations does not overlap
            m_pos_uavs_in_grid[ind] = np.random.choice(v_pos_samples,
                                                       num_uavs,
                                                       replace=False)

        v_num_ser_users = cal_num_ser_usr_mult_sols(m_pos_uavs_in_grid)

        v_uav_inds = None
        pre_num_ser_users = 0
        count_stop = 0
        for _ in range(self.max_num_gens):

            # generate the next generation
            m_new_gen = gen_new_gen(m_pos_uavs_in_grid, v_num_ser_users)

            # compute fitness values - associate users
            v_num_ser_users_new = cal_num_ser_usr_mult_sols(m_new_gen)

            if debug:
                print("Max num served users: {}".format(
                    np.max(v_num_ser_users_new)))

            # update the objective values
            if np.max(v_num_ser_users_new) > pre_num_ser_users:
                v_uav_inds = m_new_gen[np.argmax(v_num_ser_users_new)]
                max_num_ser_user = np.max(v_num_ser_users_new)
                # reset count to stop
                count_stop = 0
            else:
                count_stop += 1

            # check if all users are served
            if max_num_ser_user == num_users:
                return v_uav_inds
            if (count_stop == 5) and (max_num_ser_user < num_users):
                return None

            # update variables
            v_num_ser_users = v_num_ser_users_new
            m_pos_uavs_in_grid = m_new_gen
            pre_num_ser_users = max_num_ser_user

        return None


class FromMinRadiusPlacer(CapacityBasedPlacer):
    """ The number of UAVs is determined so that all users lie within a certain
        distance from the UAVs. The distance is initially determined from
        self.min_user_rate and reduced by `self.distance_reduce_factor`
        iteratively until the minimum rate is guaranteed. 

        Each user associates with the strongest UAV.
     """

    def __init__(self, num_radius=4, radius_discount=0.9, **kwargs):
        """ Args:

            `num_radius`: number of values for the radius to try between the
            radius determined by the rate without grid quantization and with
            grid quantization.

            `radius_discount`: if the minimum radius that guarantees the
            existence of a solution in free space is reached, then the
            subsequent attempted radii are obtained by multiplying the previous
            one by `radius_discount`.

        """
        super().__init__(**kwargs)
        assert self.max_uav_total_rate is None, "this placer cannot guarantee a max total rate per UAV"

        assert self.min_user_rate is not None
        self.num_radius = num_radius
        self.radius_discount = radius_discount

    def place(self,
              fly_grid,
              channel,
              user_coords,
              *args,
              delta_radius=.1,
              debug=0,
              **kwargs):
        """ See parent. 

            A set of radii is used determined by the distance to guarantee a
            certain capacity and the error due to the grid quantization. A
            solution is guaranteed in the worst case in free space, but larger
            radii are tried first just in case the number of UAVs can be
            reduced. 

            Args:

            `delta_radius`: the radius guaranteeing coverage is reduced by this
            amount to avoid numerical problems. 
        """

        map = channel.capacity_map(
            grid=fly_grid,
            user_coords=user_coords,
        )
        m_capacity = map.list_vals().T

        # Determine values of the radius to use
        assert self.min_user_rate > 0
        max_dist = channel.max_distance_for_rate(self.min_user_rate)
        radius_bf_quantization = np.sqrt(max_dist**2 -
                                         fly_grid.min_enabled_height**2)
        # The following is the "critical radius". Always feasible in free space.
        radius_after_quantization = radius_bf_quantization - map.grid.max_herror
        if radius_after_quantization < 0:
            raise ValueError(
                "Either the rate is too low or the grid to coarse")
        supercritical_radii = np.linspace(radius_bf_quantization,
                                          radius_after_quantization,
                                          num=self.num_radius)
        # Determine the number of subcritical radii until we reach a radius = map.grid.max_herror
        num_subc_radii = np.ceil(
            np.log(map.grid.max_herror / radius_after_quantization) /
            np.log(self.radius_discount))
        subcritical_radii = radius_after_quantization * (
            self.radius_discount**np.arange(1, num_subc_radii + 1))
        v_radius = np.concatenate([supercritical_radii, subcritical_radii])
        v_radius -= delta_radius
        # plt.plot(v_radius, "-x")
        # plt.show()

        for radius in v_radius:
            inds_uavs = self._place_given_radius(map=map,
                                                 user_coords=user_coords,
                                                 radius=radius,
                                                 debug=debug)
            if debug:
                print("radius = ", radius)
                print(
                    "non-covered users = ",
                    np.sum(
                        np.max(m_capacity[:, inds_uavs], axis=1) <
                        self.min_user_rate))
                print("num_uavs=", len(inds_uavs))

                ind_non_covered_users = np.where(
                    np.max(m_capacity[:, inds_uavs], axis=1) <
                    self.min_user_rate)[0]
                if len(ind_non_covered_users):
                    ind_user = ind_non_covered_users[0]
                    uc = user_coords[ind_user]
                    uav_coords = map.grid.list_pts()[inds_uavs]
                    hdists = np.linalg.norm(uav_coords[:, 0:2] - uc[None, 0:2],
                                            axis=1)
                    ind_nearest_uav = inds_uavs[np.argmin(hdists)]
                    coords_nearest_uav = map.grid.list_pts()[ind_nearest_uav]
                    min_hdist = np.min(hdists)
                    print("hdist to closest UAV", min_hdist)
                    print("capacity to closest UAV",
                          m_capacity[ind_user, ind_nearest_uav])
                    dbgain = channel.dbgain(uc, coords_nearest_uav)
                    channel.dbgain_to_capacity(dbgain)
                    channel.dist_to_dbgain_free_space(
                        np.linalg.norm(coords_nearest_uav - uc))

            if all(
                    np.max(m_capacity[:, inds_uavs], axis=1) >=
                    self.min_user_rate):
                return map.grid.list_pts()[inds_uavs]

        # No solution found.
        log.warning(
            f"Maximum number of iterations reached at {self.__class__.__name__} without guaranteeing a minimum rate"
        )
        return None

    # Abstract methods
    def _place_given_radius(self, map, user_coords, radius):
        """ Returns a vector of indices of the grid points where UAVs need to be
        placed. 
        """
        raise NotImplementedError


class SpiralPlacer(FromMinRadiusPlacer):
    """ Implements lyu2017mounted. MATLAB code provided by the authors."""
    _name_on_figs = "Lyu et al."

    def _place_given_radius(self, map, user_coords, radius, debug):
        """ Returns a vector of indices of the grid points where UAVs need to be
        placed. 
        """
        r = Runner("placement", "Spiral.m")
        data_in = OrderedDict()
        data_in["m_users"] = user_coords[:, 0:2].T
        data_in["radius"] = radius
        # 2 x num_uavs
        #uav_coords_2d = r.run("save", data_in)[0]
        uav_coords_2d = r.run("place", data_in)[0]

        # num_uavs x 3
        uav_coords_3d = np.concatenate(
            (uav_coords_2d.T, np.zeros((uav_coords_2d.shape[1], 1))), axis=1)

        v_uav_inds = np.sort(map.grid.nearest_inds(uav_coords_3d))

        if debug:

            def dist_to_nearest_uav(uav_coords, uc):
                dists = np.linalg.norm(uav_coords[:, 0:2] - uc[None, 0:2],
                                       axis=1)
                #dists = np.linalg.norm(uav_coords_2d.T - uc[None, 0:2], axis=1)
                return np.min(dists)

            max_min_dist = np.max(
                [dist_to_nearest_uav(uav_coords_3d, uc) for uc in user_coords])
            print("max hmin distance bf discretization: ", max_min_dist)

            uav_coords_grid = map.grid.list_pts()[v_uav_inds, :]
            max_min_dist = np.max([
                dist_to_nearest_uav(uav_coords_grid, uc) for uc in user_coords
            ])
            print("max hmin distance after discretization: ", max_min_dist)

            for uavc in uav_coords_3d:
                uavc_grid = map.grid.nearest_pt(uavc)
                print(
                    f"uav at {uavc} mapped to {uavc_grid}. HError: {np.linalg.norm(uavc[0:2]-uavc_grid[0:2])}"
                )

        return v_uav_inds


class SparseRecoveryPlacer(FromMinRadiusPlacer):
    """ Implements huang2020sparse. """
    _name_on_figs = "Huang et al."
    max_num_users = 15  # above that, it just returns None, since it is computationally too complex

    def place(self, user_coords, *args, **kwargs):
        num_users = len(user_coords)
        if num_users > self.max_num_users:
            return None
        return super().place(user_coords=user_coords, *args, **kwargs)

    def _place_given_radius(self, map, user_coords, radius, debug):
        """ Returns a vector of indices of the grid points where UAVs need to be
        placed. 
        """
        r = Runner("placement", "SparseRecoveryPlacer.m")
        data_in = OrderedDict()
        data_in["m_users"] = user_coords[:, 0:2]
        data_in["radius"] = radius
        # 2 x num_uavs
        #uav_coords_2d = r.run("save", data_in)[0]
        uav_coords_2d = r.run("place", data_in)[0]

        # num_uavs x 3
        uav_coords_3d = np.concatenate(
            (uav_coords_2d, np.zeros((uav_coords_2d.shape[0], 1))), axis=1)

        v_uav_inds = np.sort(map.grid.nearest_inds(uav_coords_3d))

        if debug:

            def dist_to_nearest_uav(uav_coords, uc):
                dists = np.linalg.norm(uav_coords[:, 0:2] - uc[None, 0:2],
                                       axis=1)
                #dists = np.linalg.norm(uav_coords_2d.T - uc[None, 0:2], axis=1)
                return np.min(dists)

            max_min_dist = np.max(
                [dist_to_nearest_uav(uav_coords_3d, uc) for uc in user_coords])
            print("max hmin distance bf discretization: ", max_min_dist)

            uav_coords_grid = map.grid.list_pts()[v_uav_inds, :]
            max_min_dist = np.max([
                dist_to_nearest_uav(uav_coords_grid, uc) for uc in user_coords
            ])
            print("max hmin distance after discretization: ", max_min_dist)

            for uavc in uav_coords_3d:
                uavc_grid = map.grid.nearest_pt(uavc)
                print(
                    f"uav at {uavc} mapped to {uavc_grid}. HError: {np.linalg.norm(uavc[0:2]-uavc_grid[0:2])}"
                )

        return v_uav_inds