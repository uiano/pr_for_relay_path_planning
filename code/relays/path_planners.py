import random
import sys
from line_profiler import profile

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from gsim_conf import use_mayavi
from common.fields import FunctionVectorField
import copy
import warnings
import heapq
import time


class PathPlanner():

    # to be overridden by subclasses
    _name_on_figs = ""

    def __init__(self,
                 environment,
                 channel,
                 pars_on_name=[],
                 name_custom=None):
        """
        
        Args:

            - `bs_loc`: (N,3)-array-like where N is the number of ground BSs.

            - `pars_on_name`: list of tuples (str_format, par_name) used by __str__. 

        """

        self._environment = environment
        self._channel = channel
        self.pars_on_name = pars_on_name
        self.name_custom = name_custom

    def _loc_at_each_time_step(self, start_loc, end_loc, samp_int,
                               max_uav_speed):
        """
            Discretizes the line segment between start_loc and end_loc into a list of locations such that the distance between two consecutive locations is smaller than or equal sampt_int * max_uav_speed.
        """

        # Find the distance that the UAV can travel in one sample interval
        dist_per_int = max_uav_speed * samp_int

        uav_loc_start = start_loc

        v_direction = end_loc - uav_loc_start
        # So that the UAV does not change its height

        dist = np.linalg.norm(np.ravel(v_direction))

        assert dist != 0
        v_direction = v_direction / dist
        num_steps = int(np.floor(dist / dist_per_int))

        # Matrix of points from uav_loc_start to the mid point of uav_loc_start and ue_loc
        l_uav_loc = [uav_loc_start]
        l_uav_loc += [
            uav_loc_start + ind_step * dist_per_int * v_direction
            for ind_step in range(1, num_steps + 1)
        ]

        return l_uav_loc

    def _path_w_takeoff(self, uav_loc_start, uav_loc_end, samp_int,
                        max_uav_speed):
        """
            Returns:
                + l_loc_vs_time: num_loc x 3, a list of locations from
                  uav_loc_start to ue_loc which consists of the locations
                  between uav_loc_start and the point above it (locations in the
                  takeoff phase) and between that point and the point above
                  ue_loc.
        """
        # Take off phase
        uav_loc_takeoff_end = np.array(
            [uav_loc_start[0], uav_loc_start[1], self.fly_height])

        l_loc_takeoff_vs_time = self._loc_at_each_time_step(
            uav_loc_start, uav_loc_takeoff_end, samp_int, max_uav_speed)

        num_takeoff_int = len(l_loc_takeoff_vs_time)

        # Flying phase
        ue_loc_above = np.array(
            [uav_loc_end[0], uav_loc_end[1], self.fly_height])
        l_loc_fly_vs_time = self._loc_at_each_time_step(
            uav_loc_takeoff_end, ue_loc_above, samp_int, max_uav_speed)

        l_loc_fly_vs_time.append(ue_loc_above)

        num_fly_int = len(l_loc_fly_vs_time)

        l_loc_vs_time = l_loc_takeoff_vs_time + l_loc_fly_vs_time

        return l_loc_vs_time, num_takeoff_int, num_fly_int

    def constrain_path_with_rmin(self, lm_path, min_uav_rate, bs_loc):
        """
            Returns a sublist of lm_path containing the num_int first entries of
            lm_path, where num_int is the largest integer for which all UAVs in
            m_pos := lm_path[ind_int] can receive a rate `min_uav_rate` from the base
            station. The latter condition means that:
            
            - capacity of the link [bs_loc -> m_pos[0]] >= num_uavs * min_uav_rate
            - capacity of the link [m_pos[0], m_pos[1]] >= (num_uavs-1) * min_uav_rate
            - ...
            - capacity of the link [m_pos[num_uavs-2], m_pos[num_uavs-1]] >= min_uav_rate
            
            where num_uavs := m_pos.shape[0].

            Returns:
                lm_path with a constrained trajectory of UAV1
        """

        if min_uav_rate is None:
            return lm_path

        # take out the trajectory of the 1st UAV.
        #m_uav1_loc = np.array([m_loc[0] for m_loc in lm_path])

        # compute the rate at each location
        if lm_path is None:
            return None

        m_uavs_rate = PathPlanner.rate_from_path(self._channel, lm_path,
                                                 min_uav_rate, bs_loc)

        l_infea_loc = list(np.where(m_uavs_rate[-1, :] < min_uav_rate)[0])
        if len(l_infea_loc) == 0:
            # no need to constrain the trajectory of the 1st UAV
            stop_ind = len(lm_path)
        else:
            stop_ind = l_infea_loc[0]

        # for ind_loc in range(stop_ind, len(lm_path)):
        #     lm_path[ind_loc][0] = lm_path[stop_ind - 1][0]

        return lm_path[:stop_ind]

    def plan_path_to_serve_static_ue(self, bs_loc, ue_loc, samp_int,
                                     max_uav_speed, uav_loc_start):
        """
        Args:

        `bs_loc`: (3,) vector

        `uav_loc_start`: Possibilities:
                    
            - (num_uavs,3) matrix with the starting location of the
        UAVs. 
        
            - (3,) vector: in this case, it is understood that all UAVS start from
        the same location `uav_loc_start`

            - None: in this case, `uav_loc_start` is set to `bs_loc`

        `ue_loc`: (3,) vector with the location of the user to be served. 

        `samp_int`: sampling interval in seconds

        `max_uav_speed`: scalar. 

        Returns:

        `lm_path` : list of configuration points. `lm_path[n]` is a matrix of
        shape num_uavs x 3 whose i-th row provides the position of the i-th UAV
        at time `n*samp_int`. It must hold that || lm_path[n][i+1] -
        lm_path[n][i] || / samp_int <= max_uav_speed. 
        
        """

    @staticmethod
    def rate_from_path(channel,
                       lm_path,
                       min_uav_rate,
                       bs_loc,
                       lm_ue_path=None):
        """
        Args:
            + lm_ue_path can be
                . None, returns the rates of the UAVs over time;

                . a (3,) vector, returns the rates of the UAVs and the rate
                of the UE over time;

                . a list of (1, 3) matrices, returns the rates of the UAVs
                and the rate of the UE in every time step.
        
        Returns:
            + m_uavs_rate: (num_uavs x num_time_step) matrix            

            + v_ue_rate: (num_time_step, ) vector if lm_ue_path is None.
        """
        if lm_path is None:
            if lm_ue_path is not None:
                return None, None

        num_time_step = len(lm_path)
        num_uavs = lm_path[0].shape[0]
        m_uavs_rate = np.zeros((num_uavs, num_time_step))
        v_user_rate = np.zeros((num_time_step, ))

        def compute_capacity(pt1, pt2):
            return channel.dbgain_to_capacity(channel.dbgain(pt1, pt2))

        def compute_rate_ue(uav2_rate, uav2_loc, ue_loc):
            return np.minimum(uav2_rate - min_uav_rate,
                              compute_capacity(uav2_loc, ue_loc))

        for ind_step in range(num_time_step):
            # Compute UAV1's rate
            m_uavs_rate[0,
                        ind_step] = compute_capacity(bs_loc,
                                                     lm_path[ind_step][0])
            for ind_uav in range(1, num_uavs):
                if m_uavs_rate[ind_uav - 1, ind_step] >= min_uav_rate:

                    # Compute rate UAV (i-1) -> UAV i
                    m_uavs_rate[ind_uav, ind_step] = np.minimum(
                        m_uavs_rate[ind_uav - 1, ind_step] - min_uav_rate,
                        compute_capacity(lm_path[ind_step][ind_uav - 1],
                                         lm_path[ind_step][ind_uav]))

                else:
                    m_uavs_rate[ind_uav, ind_step] = 0

            # Compute users's rate
            if type(lm_ue_path) is np.ndarray:
                if m_uavs_rate[-1, ind_step] >= min_uav_rate:
                    v_user_rate[ind_step] = compute_rate_ue(
                        m_uavs_rate[-1, ind_step], lm_path[ind_step][-1],
                        lm_ue_path)
                else:
                    v_user_rate[ind_step] = 0
            elif type(lm_ue_path) is list:
                v_user_rate[ind_step] = compute_rate_ue(
                    m_uavs_rate[-1, ind_step], lm_path[ind_step][-1],
                    lm_ue_path[ind_step][0])
            else:
                pass

        if lm_ue_path is not None:
            return m_uavs_rate, v_user_rate
        else:
            return m_uavs_rate

    @staticmethod
    def conf_pt_to_ue_rate(channel, conf_pt, min_uav_rate, bs_loc, ue_loc):
        _, v_ue_rate = PathPlanner.rate_from_path(channel, [conf_pt],
                                                  min_uav_rate, bs_loc, ue_loc)
        return v_ue_rate[0]

    @staticmethod
    def conf_pt_to_is_ue_rate_gt_min(channel,
                                     conf_pt,
                                     min_uav_rate,
                                     min_ue_rate,
                                     bs_loc,
                                     ue_loc,
                                     v_los=None):
        """Returns True iff the UE rate associated with conf_pt `conf_pt` is
        greater than or equal to `min_uav_rate`
        
        Args:
            `v_los` is a vector of length num_uavs + 1 whose n-th entry is True
            if the n-th link is known to have LOS. This avoids integration. If
            None, it is assumed that `v_los=[False, False, ..., False]`
        ."""

        num_uavs = conf_pt.shape[0]
        if v_los is None:
            v_los = [False] * (num_uavs + 1)
        conf_pt_with_ue = np.concatenate((conf_pt, ue_loc[None, :]), axis=0)

        def compute_capacity(pt1, pt2, b_los):
            if b_los:
                return channel.dbgain_to_capacity(
                    channel.dbgain_free_space(pt1, pt2))
            else:
                return channel.dbgain_to_capacity(channel.dbgain(pt1, pt2))

        # Compute UAV1's rate
        rate_so_far = compute_capacity(bs_loc, conf_pt[0], v_los[0])
        for ind_uav in range(1, num_uavs + 1):

            if rate_so_far - min_uav_rate < min_ue_rate:
                return False

            rate_uav_link = compute_capacity(conf_pt_with_ue[ind_uav - 1],
                                             conf_pt_with_ue[ind_uav],
                                             v_los[ind_uav])
            # Rate of the ind_uav-th UAV (or UE if ind_uav==num_uavs)
            rate_so_far = np.minimum(rate_so_far - min_uav_rate, rate_uav_link)

        return rate_so_far >= min_ue_rate

    @staticmethod
    def avg_dist_from_path(lm_path):

        num_time_slots = len(lm_path)
        v_avg_dist = np.zeros((num_time_slots, ))
        num_uavs = lm_path[0].shape[0]
        for ind_conf_pt in range(1, num_time_slots):

            avg_dist = np.sum(
                np.linalg.norm(lm_path[ind_conf_pt] - lm_path[ind_conf_pt - 1],
                               axis=1)) / num_uavs
            v_avg_dist[ind_conf_pt] = v_avg_dist[ind_conf_pt - 1] + avg_dist

        return v_avg_dist

    @property
    def name(self):
        if self.name_custom is not None:
            return self.name_custom
        return self.__class__.__name__

    @property
    def name_on_figs(self):
        if self.name_custom is not None:
            return self.name_custom

        if self._name_on_figs:
            return self._name_on_figs
        else:
            return self.name

    def time_to_min_ue_rate(self, lm_path, min_uav_rate, min_ue_rate, samp_int,
                            bs_loc, ue_loc):
        """
        Returns:             
            + The first time instant when user's rate is greater than min_uav_rate, and
            + None, otherwise.
            
        """

        _, v_ue_rate = self.rate_from_path(self._channel,
                                           lm_path,
                                           min_uav_rate,
                                           bs_loc,
                                           lm_ue_path=ue_loc)
        for ind_rate in range(len(v_ue_rate)):
            if v_ue_rate[ind_rate] >= min_ue_rate:
                return ind_rate * samp_int

        return None

    def time_to_los(self, lm_path, samp_int, ue_loc):
        """
        Returns:             
            + The first time instant when the last uav in lm_path is in LoS with ue_loc and the capacity between the uav and the user at that time.
            + None, otherwise.
        
        Assumption: There is always connectivity from the base station to the last uav in configuration points in lm_path.
            
        """

        def compute_capacity(pt1, pt2):
            return self._channel.dbgain_to_capacity(
                self._channel.dbgain(pt1, pt2))

        m_last_uav_trajectory = np.array(lm_path)[:, -1, :]

        for ind_loc in range(len(m_last_uav_trajectory)):
            if self.are_in_los(ue_loc, m_last_uav_trajectory[ind_loc]):

                return [
                    np.round(ind_loc * samp_int, 1),
                    compute_capacity(ue_loc, m_last_uav_trajectory[ind_loc])
                ]

        return None

    def are_in_los(self, pt1, ref_pts, integral_mode="c"):
        """
        Args:

        ref_pts is either a 3D point or a list/np.array of 3D points.

        Returns True iff `pt1` is in LOS with at least one of the points in ref_pts.
        """

        ref_pts = np.array(ref_pts)
        if ref_pts.ndim == 1:
            ref_pts = ref_pts[None, :]

        for ref_pt in ref_pts:
            if self._environment.slf.line_integral(pt1,
                                                   ref_pt,
                                                   mode=integral_mode)[0] == 0:
                return True
        return False

    def compute_capacity(self, pt1, pt2):
        return self._channel.dbgain_to_capacity(self._channel.dbgain(pt1, pt2))

    def is_at_least_rate(self, pt1, ref_pts, min_rate):
        """
        Args:

        ref_pts is either a 3D point or a list/np.array of 3D points.

        Returns True iff `pt1` has at least min_rate with at least one of the points in ref_pts.
        """

        ref_pts = np.array(ref_pts)
        if ref_pts.ndim == 1:
            ref_pts = ref_pts[None, :]

        for ref_pt in ref_pts:
            if self.compute_capacity(pt1, ref_pt) >= min_rate:
                return True
        return False

    @staticmethod
    def combine_paths(lm_path_1, lm_path_2):
        """
        Repeats the last entry of the shortest of these two lists to make both
        lists of the same length. Then, it concatenates the entries of both
        lists into a single list. 

        """

        if lm_path_1 is None or lm_path_2 is None:
            return None

        def complete(lm, target_len):
            num_extra_elements = target_len - len(lm)
            return lm + [lm[-1]] * num_extra_elements

        l1, l2 = len(lm_path_1), len(lm_path_2)
        if l1 < l2:
            lm_path_1 = complete(lm_path_1, l2)
        elif l2 < l1:
            lm_path_2 = complete(lm_path_2, l1)

        return [
            np.concatenate((lm_path_1[ind], lm_path_2[ind]), axis=0)
            for ind in range(len(lm_path_1))
        ]

    @staticmethod
    def get_ue_rates_all_los_conf_pts(environment, channel, min_uav_rate,
                                      bs_loc, ue_loc):
        """
        It returns a list of the ue rates of the configuration points that have LOS at all links. 
        
        """
        pp = PathPlanner(environment=environment, channel=channel)
        fly_grid = pp._environment.fly_grid
        m_fly_pts = fly_grid.list_pts()

        m_fly_pts_uav_1 = [
            v_fly_pt for v_fly_pt in m_fly_pts
            if pp.are_in_los(v_fly_pt, bs_loc)
        ]

        m_fly_pts_uav_2 = [
            v_fly_pt for v_fly_pt in m_fly_pts
            if pp.are_in_los(v_fly_pt, ue_loc)
        ]

        l_conf_pts = [
            np.concatenate((v_pt_uav_1[None, :], v_pt_uav_2[None, :]), axis=0)
            for v_pt_uav_1 in m_fly_pts_uav_1 for v_pt_uav_2 in m_fly_pts_uav_2
        ]

        return [
            pp.conf_pt_to_ue_rate(channel=channel,
                                  conf_pt=conf_pt,
                                  min_uav_rate=min_uav_rate,
                                  bs_loc=bs_loc,
                                  ue_loc=ue_loc) for conf_pt in l_conf_pts
        ]

    def __str__(self):
        str_pars = ""
        if len(self.pars_on_name):
            l_pars = [
                par_format.format(getattr(self, par_name))
                for par_format, par_name in self.pars_on_name
            ]
            str_pars = " (" + ", ".join(l_pars) + ")"
        return self.name_on_figs + str_pars

    @staticmethod
    def resample(lm_path_coarse, samp_int, max_speed):

        if lm_path_coarse is None:
            return None

        # a list of time traveling from one conf_pt to another.
        l_time_arrive_at_conf_pt = [0]

        for ind_waypt in range(1, len(lm_path_coarse)):

            l_time_arrive_at_conf_pt.append(
                l_time_arrive_at_conf_pt[-1] + np.max(
                    np.linalg.norm(lm_path_coarse[ind_waypt] -
                                   lm_path_coarse[ind_waypt - 1],
                                   axis=1)) / max_speed)

        sampling_instant = 0
        lm_path_resampled = [lm_path_coarse[0]]
        while sampling_instant < l_time_arrive_at_conf_pt[-1] - samp_int:

            sampling_instant += samp_int
            # find n s.t. t_n <= sampling_instant <t_n + 1
            ind_prev_waypt = np.where(
                np.array(l_time_arrive_at_conf_pt) <= sampling_instant)[0][-1]
            m_conf_pt = 1 / (l_time_arrive_at_conf_pt[ind_prev_waypt + 1] -
                             l_time_arrive_at_conf_pt[ind_prev_waypt]) * (
                                 lm_path_coarse[ind_prev_waypt + 1] -
                                 lm_path_coarse[ind_prev_waypt]) * (
                                     sampling_instant -
                                     l_time_arrive_at_conf_pt[ind_prev_waypt]
                                 ) + lm_path_coarse[ind_prev_waypt]

            lm_path_resampled.append(m_conf_pt)

        if np.linalg.norm(lm_path_coarse[-1] - lm_path_resampled[-1]) != 0:
            lm_path_resampled.append(lm_path_coarse[-1])

        return np.round(lm_path_resampled, 3)

    @staticmethod
    def get_eind_first_enode_each_block(ll_uav1_fea_inds_at_each_uav2_wpts):
        l_num_nodes_each_block = [
            len(l_inds) for l_inds in ll_uav1_fea_inds_at_each_uav2_wpts
        ]

        # The i-th entry of the following list is the extended node index (eind)
        # of the first enode in the i-th block. The last entry is the total
        # number of enodes.
        v_eind_first_enode_each_block = np.cumsum([0] + l_num_nodes_each_block)
        return v_eind_first_enode_each_block

    @staticmethod
    def get_shortest_path(m_cost,
                          ind_node_start,
                          ind_nodes_end,
                          implementation='heap',
                          num_nodes_per_block=None,
                          debug=False):
        """
        Args:

            - m_cost is an num_nodes x num_nodes matrix whose (m,n)-th entry is
              the cost of going from node m to node n. It is np.Inf if m and n
              are not connected. 

            - ind_nodes_end: set/list of indices of the possible destinations.
              If `ind_nodes_end` is an integer, then it is understood as
              {ind_nodes_end}.

            - If implementation == 'heap' and num_nodes_per_block is
                . None, the algorithm will search over all nodes in m_cost[n, :]
                to find the neighbors of node n.

                . a scalar, it is assumed that 1) the blocks are square and have
                the same size; 2) only the block diagonal above the main block
                diagnonal of m_cost contains entries different from infinity.
                This means that the set of neighbors of a node in block n are in
                block n+1.
            
                . a list, block n has num_nodes_per_block[n] nodes. The set of
                neighbors of a node in block n are in block n+1.
                
        Returns:

        l_path: list of indices of nodes forming the shortest path where 

            - l_path[0] = ind_node_start
            - l_path[-1] = ind_node_end
            - sum_i m_cost[ l_path[i] , l_path[i+1] ] is minimum

            l_path is None if there is no path between the nodes with indices
            ind_node_start and ind_node_end. 

        If ind_nodes_end is empty, the function returns None.

        """

        def get_shortest_path_noheap(m_cost, ind_node_start, ind_nodes_end):

            num_nodes = m_cost.shape[0]

            if (type(ind_nodes_end) == int) or (type(ind_nodes_end)
                                                == np.int64):
                ind_nodes_end = {ind_nodes_end}
            else:  # else, it is a set or a list
                ind_nodes_end = set(ind_nodes_end)
            if len(ind_nodes_end) == 0:
                return None

            # Find the node with minimum distance to the source and the node
            # immediately before it in the shortest path.
            # @profile
            def f_node_to_add(v_distances, l_nodes_added_so_far):
                """ 
                    The i-th entry of v_distances contains the length of the best
                    path known so far between the starting node and the i-th node. 

                    Among the nodes not yet included in the shortest path, find 

                        1. the node with the minimum distance so far to the
                            starting node 
                        
                        2. and the node in the shortest path that the newly added
                        node should go through to go towards the starting point.
                """

                # Find the node that has not been added yet with minimum distance to
                # the starting node.
                v_inds = np.where(np.logical_not(l_nodes_added_so_far))[0]
                node_to_add = v_inds[np.argmin(v_distances[v_inds])]

                # Among the nodes in l_nodes_added_so_far and adjacent to node
                # node_to_add, we choose the node n such that dist(starting_node, n)
                # + dist(n, node_to_add) +  is minimum.
                if not sum(l_nodes_added_so_far):
                    # First iteration
                    node_to_go = node_to_add
                else:
                    # TODO: simplify

                    # In l_nodes_added_so_far, find the nodes that are adjacent to
                    # node node_to_add
                    # l_candidate_prev_nodes = []
                    # for ind_node in range(num_nodes):
                    #     if m_cost[ind_node,
                    #               node_to_add] < np.inf and l_nodes_added_so_far[
                    #                   ind_node] == True:
                    #         # why condition after and? --> bc. any node N that has
                    #         # not been added has a greater distance to the starting
                    #         # point than `node_to_add` because of the way
                    #         # `node_to_add` was chosen and by the fact that to reach
                    #         # N, one must necessarily go through one added node and
                    #         # (possibly) one neighbor of an added node.
                    #         l_candidate_prev_nodes.append(ind_node)

                    l_candidate_prev_nodes = np.where(
                        np.logical_and(m_cost[:, node_to_add] != np.inf,
                                       l_nodes_added_so_far))[0]

                    # Choose the previous node in l_candidate_prev_nodes that
                    # results in the shortest path to `node_to_add`.
                    l_dist_to_source = []
                    for ind_node in l_candidate_prev_nodes:
                        l_dist_to_source.append(m_cost[ind_node, node_to_add] +
                                                v_distances[ind_node])

                    node_to_go = l_candidate_prev_nodes[np.argmin(
                        l_dist_to_source)]

                return node_to_add, node_to_go

            def f_path_from_source(ind_node_start, l_nearest_node_in_path,
                                   node_on_path):
                """ 
                    Find the path from each node to the starting node. 

                    Args: 

                    `l_nearest_node_in_path`: list where the i-th element is the
                    index of the 2nd node in the shortest path from the
                    i-th node to the starting node.

                    Returns: 

                    - list where the i-th element is a path from the
                        i-th node to the starting point. 
                        
                """

                # Find the path from node ind to the starting node.
                l_path_to_source = []
                while ind_node_start != node_on_path:
                    l_path_to_source.append(node_on_path)
                    node_on_path = l_nearest_node_in_path[node_on_path]

                # Finally, add the starting node to the path
                l_path_to_source.append(node_on_path)
                l_path_to_source.reverse()

                return l_path_to_source

            def update_distances_from_source(v_distances, node_to_add,
                                             l_nodes_added_so_far):
                # Update distances from the source to all nodes that
                #
                #       + are not in l_nodes_added_so_far, AND
                #
                #       + are adjacent to node_to_add (the newly added node)
                #
                # if their distances to the starting node become smaller when
                # going through node_to_add.
                # for node_compared in range(num_nodes):

                l_dist_node_to_add = v_distances[node_to_add]
                for node_compared in np.where(
                        m_cost[node_to_add] != np.inf)[0]:
                    cost_now = m_cost[node_to_add, node_compared]

                    if l_nodes_added_so_far[node_compared]:
                        continue
                    if v_distances[
                            node_compared] <= l_dist_node_to_add + cost_now:
                        continue

                    v_distances[node_compared] = v_distances[
                        node_to_add] + m_cost[node_to_add, node_compared]

                return v_distances

            # Function that implements Dijkstra's single source
            # shortest path algorithm for a graph represented
            # using an adjacency matrix
            def f_dijkstra(ind_node_start, set_end_nodes):
                """"
                
                Property: if node N_1 is added before node N_2, then N_1 is closer
                to the starting point than N_2. This is because a node N is added by
                choosing the one from the list of candidates, which comprises those
                nodes for which l_distances[ind_node] < sys.maxsize and it has not
                been added, that has lowest temporary (not final) distance to the
                starting point. That temporary distance equals the shortest distance
                to an added node plus the distance from that added node to N. That
                distance is a lower bound for the distance of all nodes that have
                not been added to the starting node because (i) going to any node
                that has not been added through the shortest path necessarily
                involves going through or to one of the candidate nodes, (ii) this
                candidate node is connected to an added node and therefore its
                temporary distance equals its final distance, and (iii) such a
                shortest path is clearly longer than the path to N. 
                """

                # At every iteration, l_distances[i] contains the length of the
                # shortest path known so far from the starting node to the i-th
                # node. Initially, all elements are infinity except the
                # ind_node_start-th element, which is zero.
                v_distances = np.array(
                    [np.Inf] *
                    num_nodes)  #l_distances = [sys.maxsize] * num_nodes
                v_distances[ind_node_start] = 0

                # The following list is an indicator of those nodes for which
                # we know, at every iteration, the shortest path from the
                # starting node. Specifically, throughout the algorithm, the
                # i-th entry of the following list is True if l_distances[i]
                # contains the length of the actual shortest path between the
                # starting point and the i-th node. When it is False,
                # l_distances[i] can decrease as the iterations go by and,
                # therefore, better paths to reach node i can be discovered.
                l_nodes_added_so_far = [False] * num_nodes

                # l_nodes_to_go[i] is False when the shortest path to node i has
                # not been found yet. It is j if node j is the node immediately
                # before node i in the shortest path to i.
                l_nodes_to_go = [False] * num_nodes

                for _ in range(num_nodes):
                    # At every iteration, we discover the shortest path to one
                    # node, namely the one with index `node_to_add`.

                    # Pick the node that has the minimum distance to the
                    # starting point and that was not added yet. `node_to_go` is
                    # the previous node on the path from the starting node to
                    # `node_to_add`.
                    node_to_add, node_to_go = f_node_to_add(
                        v_distances, l_nodes_added_so_far)

                    l_nodes_added_so_far[node_to_add] = True
                    l_nodes_to_go[node_to_add] = node_to_go

                    # Stop if the shortest path to one of the nodes in
                    # `set_end_nodes` has been found.
                    if node_to_add in set_end_nodes:
                        break

                    v_distances = update_distances_from_source(
                        v_distances, node_to_add, l_nodes_added_so_far)

                    # The following condition holds iff there are no adjacent
                    # nodes to the nodes added so far and there are still nodes
                    # that have not been added. Due to the stopping condition
                    # above (break), node `ind_node_end` has not been added.
                    # Thus, there is no path towards that node.
                    if (not all(l_nodes_added_so_far)
                            == True) and (list(v_distances < sys.maxsize)
                                          == l_nodes_added_so_far):
                        if debug:
                            print(
                                "####### The start and end nodes are not connected! #######"
                            )
                        return None

                if node_to_add not in set_end_nodes:
                    """This may mean that set_end_nodes does not contain any node in
                    the graph."""
                    raise ValueError("The stopping criterion was not met")

                # Find the path from the starting node to the ending node
                return f_path_from_source(ind_node_start, l_nodes_to_go,
                                          node_to_add)

            return f_dijkstra(ind_node_start, ind_nodes_end)

        def get_shortest_path_heap(m_cost, ind_node_start, ind_nodes_end):

            if (type(ind_nodes_end) == int) or (type(ind_nodes_end)
                                                == np.int64):
                ind_nodes_end = {ind_nodes_end}
            else:  # else, it is a set or a list
                ind_nodes_end = set(ind_nodes_end)
            if len(ind_nodes_end) == 0:
                return None

            if num_nodes_per_block is not None:
                if type(num_nodes_per_block) == int:
                    l_num_nodes_per_block = [num_nodes_per_block] * int(
                        np.ceil(m_cost.shape[0] / num_nodes_per_block))

                else:
                    l_num_nodes_per_block = num_nodes_per_block

                num_block = len(l_num_nodes_per_block)
                v_eind_first_enode_each_block = np.cumsum(
                    [0] + l_num_nodes_per_block)

            def dijkstra(m_cost, start_node, end_node):

                # Initialize distances and visited set
                v_distances = np.full(len(m_cost), np.inf)
                v_distances[start_node] = 0
                visited = set()
                l_node_prev = np.full(len(m_cost), np.nan)
                l_node_prev[start_node] = start_node

                # Create priority queue with (distance, node) tuples
                priority_queue = [(0, start_node)]

                def get_neighbors(current_node):
                    """ 
                        Returns a list of tuples (neighbor, distance) of the
                        connected neighbors of current_node.
                    """

                    if num_nodes_per_block is not None:

                        # ind_block = int(
                        #     np.floor(current_node / num_nodes_per_block))
                        ind_block = np.where(
                            current_node -
                            v_eind_first_enode_each_block < 0)[0][0] - 1

                        if ind_block >= num_block - 1:
                            start_ind = v_eind_first_enode_each_block[-2]
                            end_ind = v_eind_first_enode_each_block[-1]
                        else:
                            start_ind = v_eind_first_enode_each_block[ind_block
                                                                      + 1]
                            end_ind = v_eind_first_enode_each_block[ind_block +
                                                                    2]

                    else:
                        start_ind = 0
                        end_ind = m_cost.shape[1]

                    l_neighbors = []
                    for ind in range(start_ind, end_ind):
                        if m_cost[current_node, ind] < np.inf:
                            l_neighbors.append((ind, m_cost[current_node,
                                                            ind]))

                    return l_neighbors

                while priority_queue:
                    # Pop node with smallest distance
                    current_distance, current_node = heapq.heappop(
                        priority_queue)

                    # Skip if node has already been visited
                    if current_node in visited:
                        continue

                    # Update distances for neighboring nodes
                    l_neighbors = get_neighbors(current_node)
                    for neighbor, distance in l_neighbors:
                        distance = current_distance + distance
                        if distance < v_distances[neighbor]:
                            v_distances[neighbor] = distance
                            l_node_prev[neighbor] = current_node
                            heapq.heappush(priority_queue,
                                           (distance, neighbor))

                    # Mark node as visited
                    visited.add(current_node)
                    if current_node in end_node:
                        break

                if visited.intersection(end_node) == set():
                    if debug:
                        print(
                            "####### The start and end nodes are not connected! #######"
                        )
                    return None

                # Find path from start to end_node
                lm_path = [current_node]
                next_node = int(l_node_prev[current_node])
                while next_node != start_node:
                    lm_path.append(next_node)
                    next_node = int(l_node_prev[next_node])
                lm_path.append(start_node)

                if lm_path is not None:
                    lm_path.reverse()
                return lm_path

            return dijkstra(m_cost, ind_node_start, ind_nodes_end)

        if implementation == 'noheap':
            return get_shortest_path_noheap(m_cost, ind_node_start,
                                            ind_nodes_end)
        elif implementation == 'heap':
            return get_shortest_path_heap(m_cost, ind_node_start,
                                          ind_nodes_end)
        else:
            raise NotImplementedError

    @staticmethod
    def animate_ue_rate_vs_time(min_ue_rate,
                                samp_int,
                                lv_ue_rates_vs_time,
                                l_label,
                                legend_loc=1):
        """
        Args:
            + lv_ue_rates_vs_time: list of vectors of UE rates vs. time. Element n of vector i is the user rate at time step n of benchmark i. Different vectors may have different lengths.
        
        Return:
            + An animation ploting the UE rates vs. time.
        """

        def get_plot_params():
            max_num_samples = np.max(
                [len(v_ue_rates) for v_ue_rates in lv_ue_rates_vs_time])

            # extend the vectors to have the same length
            for ind, v_rates in enumerate(lv_ue_rates_vs_time):
                lv_ue_rates_vs_time[ind] = np.array(
                    list(v_rates) + [v_rates[-1]] *
                    (max_num_samples - len(v_rates)))

            m_ue_rates_vs_time = np.array(lv_ue_rates_vs_time)

            # get max rate over time to set the y-axis limit
            v_max_rates = np.zeros((max_num_samples, ))
            max_rate_so_far = 0
            for ind in range(max_num_samples):
                max_rate_so_far = np.maximum(
                    max_rate_so_far, np.max(m_ue_rates_vs_time[:, ind]))
                v_max_rates[ind] = 1.2 * max_rate_so_far

            return max_num_samples, v_max_rates

        max_num_samples, v_max_rates = get_plot_params()

        v_min_ue_rate = min_ue_rate * np.ones((max_num_samples, ))
        linewidth = 1.5

        x = []
        ll_rates = [[], [], [], []]
        l_min_ue_rate = []

        fig, ax = plt.subplots()

        # function that draws each frame of the animation
        def animate(ind_step):

            x.append(ind_step * samp_int)
            l_min_ue_rate.append(v_min_ue_rate[ind_step])

            ax.clear()

            for ind in range(len(lv_ue_rates_vs_time)):
                ll_rates[ind].append(lv_ue_rates_vs_time[ind][ind_step])
                ax.plot(x,
                        ll_rates[ind],
                        linewidth=linewidth,
                        label=l_label[ind])

            ax.plot(x,
                    l_min_ue_rate,
                    linestyle='dashed',
                    linewidth=linewidth,
                    color='black',
                    label='Target UE rate [Mbps]')
            ax.set_xlim([0, ind_step * samp_int])
            ax.set_ylim([0, v_max_rates[ind_step]])
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('UE Rate [Mbps]')
            ax.grid()
            ax.legend(loc=legend_loc)

        # run the animation
        ani = animation.FuncAnimation(fig,
                                      animate,
                                      frames=max_num_samples,
                                      interval=100,
                                      repeat=False)

        plt.show()

        return ani


class SingleRelayMidpointPathPlanner(PathPlanner):
    """ 
    One UAV takes off vertically to an altitude of self.fly_height. Then it flies towards the midpoint between the user and the BS at the same height. 

    The UAV stops if the rate between that UAV and the BS reaches min_uav_rate.
    
    """

    _name_on_figs = "Benchmark 1"

    def __init__(self, fly_height=40, min_uav_rate=None, **kwargs):

        super().__init__(**kwargs)
        self.fly_height = fly_height
        self.min_uav_rate = min_uav_rate

    def plan_path_to_serve_static_ue(self,
                                     bs_loc,
                                     ue_loc,
                                     samp_int,
                                     max_uav_speed,
                                     uav_loc_start=None):

        if uav_loc_start is None:
            uav_loc_start = bs_loc

        print(f'    o {self.name}')

        # trajectory without min_uav_rate
        l_loc_vs_time, _, _ = self._path_w_takeoff(uav_loc_start,
                                                   (bs_loc + ue_loc) / 2,
                                                   samp_int, max_uav_speed)
        lm_path_coarse = [loc[None, :] for loc in l_loc_vs_time]

        # lm_path_coarse = self.constrain_path_with_rmin(lm_path_coarse,
        #                                                self.min_uav_rate,
        #                                                bs_loc)

        lm_path = self.resample(lm_path_coarse, samp_int, max_uav_speed)

        return lm_path


class TwoRelaysAbovePathPlanner(PathPlanner):

    _name_on_figs = "Benchmark 2"

    def __init__(self, fly_height=40, min_uav_rate=None, **kwargs):

        super().__init__(**kwargs)
        self.fly_height = fly_height
        self.min_uav_rate = min_uav_rate

    def plan_path_to_serve_static_ue(self,
                                     bs_loc,
                                     ue_loc,
                                     samp_int,
                                     max_uav_speed,
                                     uav_loc_start=None):

        if uav_loc_start == None:
            uav_loc_start = bs_loc

        print(f'    o {self.name}')

        def plan_path_without_rmin():
            l_loc_vs_time, num_takeoff_int, num_fly_int = self._path_w_takeoff(
                uav_loc_start, ue_loc, samp_int, max_uav_speed)

            one_third_bs2ue = int(np.ceil((num_fly_int / 3)))
            num_int_uav1 = num_takeoff_int + one_third_bs2ue - 1
            num_int_uav2 = num_takeoff_int + 2 * one_third_bs2ue - 1

            lm_path = []

            for ind_time in range(num_int_uav2):
                if ind_time < num_int_uav1:
                    uav1_loc = l_loc_vs_time[ind_time]
                else:
                    uav1_loc = l_loc_vs_time[num_int_uav1]

                uav2_loc = l_loc_vs_time[ind_time]

                lm_path.append(np.array([uav1_loc, uav2_loc]))

            return lm_path

        # at this point, we have the path without the constraint on min_uav_rate
        lm_path_coarse = plan_path_without_rmin()

        # lm_path_coarse = self.constrain_path_with_rmin(lm_path_coarse,
        #    self.min_uav_rate,
        #    bs_loc)

        lm_path = self.resample(lm_path_coarse, samp_int, max_uav_speed)

        return lm_path


class UniformlySpreadRelaysPathPlanner(PathPlanner):
    """ Benchmark 3"""

    _name_on_figs = "Benchmark 3"

    def __init__(self, num_uavs=4, min_uav_rate=None, fly_height=40, **kwargs):

        super().__init__(**kwargs)
        self.num_uavs = num_uavs
        self.min_uav_rate = min_uav_rate
        self.fly_height = fly_height

    def plan_path_to_serve_static_ue(self,
                                     bs_loc,
                                     ue_loc,
                                     samp_int,
                                     max_uav_speed,
                                     uav_loc_start=None):

        if uav_loc_start == None:
            uav_loc_start = bs_loc

        print(f'    o {self.name}')

        def plan_path_without_rmin():

            lm_path = []

            v_is_update = [True] * self.num_uavs

            l_loc_vs_time, num_takeoff_int, num_fly_int = self._path_w_takeoff(
                uav_loc_start, ue_loc, samp_int, max_uav_speed)

            m_uav_loc = np.array(l_loc_vs_time)
            total_time_steps = num_takeoff_int + num_fly_int

            dis_in_time_steps_btw_uav = int(
                np.ceil(num_fly_int / (self.num_uavs - 1)))

            if m_uav_loc.shape[0] == 2:
                m_uav_des = m_uav_loc
            else:
                m_uav_des = m_uav_loc[
                    num_takeoff_int::dis_in_time_steps_btw_uav - 1]

            # Initialize: place all UAVs above the BS
            lm_path.append(
                np.tile(np.array([m_uav_loc[0]]), (self.num_uavs, 1)))

            # Update the locations of the UAVs at each time step
            for ind_step in range(1, total_time_steps):

                # If we put the following line outside of the for loop for ind_step, the results of lm_path will be wrong. Do not know why.
                m_uavs_loc_temp = np.zeros((self.num_uavs, 3))

                for ind_uav in range(self.num_uavs):

                    if v_is_update[ind_uav] == True:
                        m_uavs_loc_temp[ind_uav] = m_uav_loc[ind_step]

                        # Check if a UAV arrives at its destination
                        if all(m_uavs_loc_temp[ind_uav] == m_uav_des[ind_uav]):
                            v_is_update[ind_uav] = False

                    else:
                        m_uavs_loc_temp[ind_uav] = lm_path[ind_step -
                                                           1][ind_uav]

                lm_path.append(m_uavs_loc_temp)

            return lm_path

        lm_path_coarse = plan_path_without_rmin()

        if self.min_uav_rate is not None:
            lm_path_coarse = self.constrain_path_with_rmin(
                lm_path_coarse, self.min_uav_rate, bs_loc)

        lm_path = self.resample(lm_path_coarse, samp_int, max_uav_speed)

        return lm_path

    def plan_path_to_serve_moving_ue(self, bs_loc, lm_ue_path):
        """
        Let 
            - l_fly_levels be the list of the z_coordinates of the fly levels
              sorted in the ascending order. l_fly_levels[0] is the fly level
              closest to the ground.

            - m_takeoff_path be the matrix of size (len(l_fly_levels), 3) whose
              i-th row is the location of the nearest grid point to the BS at
              the fly level with height l_fly_levels[i].

        The grid points whose height is lower than l_fly_levels[-1] are
        disabled.
        
        Returns: lm_path: list of 2 x 3 matrices. lm_path[n] = [loc_uav1[n],
        loc_uav2[n]] presents the locations of the two UAVs at time step n. If

            - n <= len(l_fly_levels), then loc_uav1 = loc_uav2 =
                m_takeoff_path[n].
            
            - n > len(l_fly_levels), then 
                . loc_uav1[n] = m_takeoff_path[-1] 

                . loc_uav2[n] is the adjacent grid point of loc_uav2[n-1]
                nearest to the UE. Noted that the grid points whose height is
                lower than l_fly_levels[-1] are disabled.
        """
        print(f'    o {self.name}')

        fly_grid = self._environment.fly_grid.clone()
        # compute the number of fly levels
        m_grid_pts = fly_grid.list_pts()
        l_fly_levels = list(set(m_grid_pts[:, 2]))
        l_fly_levels.sort()

        # find the nearest grid point to the BS
        m_takeoff_path = np.tile(fly_grid.nearest_pt(bs_loc),
                                 (len(l_fly_levels), 1))
        m_takeoff_path[:, 2] = l_fly_levels
        m_takeoff_path = np.concatenate((bs_loc[
            np.newaxis,
        ], m_takeoff_path),
                                        axis=0)

        # disable the grid points that are lower than l_fly_levels[-1]
        def is_at_least_height(pt, height):
            return pt[2] >= height

        fly_grid.disable_by_indicator(
            lambda pt: not is_at_least_height(pt, l_fly_levels[-1]))
        l_grid_pts_on_top = fly_grid.list_pts()

        lm_path = []
        for ind_loc, m_ue_loc in enumerate(lm_ue_path):

            if ind_loc <= len(l_fly_levels):
                m_loc_uav1 = m_takeoff_path[ind_loc][None, :]
                m_loc_uav2 = m_takeoff_path[ind_loc][None, :]
            else:

                # find the adjacent grid points of lm_path[-1]
                l_adj_grid_pts = l_grid_pts_on_top[list(
                    fly_grid.get_inds_adjacent(lm_path[-1][1, :],
                                               l_grid_pts_on_top))]

                m_loc_uav2 = l_adj_grid_pts[np.argmin([
                    np.linalg.norm(pt - m_ue_loc) for pt in l_adj_grid_pts
                ])][None, :]

            lm_path.append(np.concatenate((m_loc_uav1, m_loc_uav2), axis=0))

        return lm_path


class RandomRoadmapPathPlanner(PathPlanner):

    _name_on_figs = "PRFI"  # probabilistic path planner

    integral_mode = 'c'  # Can be 'c' or 'python'

    def __init__(self,
                 num_uavs=None,
                 num_nodes=None,
                 max_num_neighbors=None,
                 step_connectivity=None,
                 min_uav_rate=None,
                 fly_height=None,
                 des_coor=None,
                 mode_draw_conf_pt=None,
                 mode_connect=None,
                 destination=None,
                 min_ue_rate=None,
                 b_conf_pts_meet_min_ue_rate=False,
                 ue_rate_below_target_penalty=None,
                 b_tentative=None,
                 **kwargs):
        """
        Assumption: the grid spacing of self._environment.fly_grid must be
        smaller than the side of all buildings. 


        Args:

            - `num_nodes`: If `mode_draw_conf_pt=="feasible"`, this is the 
               number of nodes for the RandomRoadmaps graph besides those of the
               feasible trajectory. For other values of `mode_draw_conf_pt`, it
               is the total number of nodes in that graph. 

            - `step_connectivity`: Two nearest neighbors are declared connected
              if a set of intermediate points between them spaced by
              `step_connectivity` is contained in Q_free. By default, it is set
              large enough so that if the UAVs move to adjacent grid points and
              have LOS at the beginning and at the end, then no intermediate
              points are checked. This is in line with the above assumption. It
              may happen that although the starting and end points are
              connected, some points in between may not due to a corner in a
              building. This in practice can be avoided just by adopting a
              denser grid or by enlarging the building model. 

            - 'fly_height': to have the same interfact with other planners.

            - `des_coor`: Currently, the destination set is defined as the set
              of configuration points whose UAVs have the x and y coordinates
              greater than des_coor. In the next steps, we may have different
              criteria to define different destination sets and need to refactor
              this argument and its caller.
        
            - 'destination': 
                - if "los", the trajectory is planned so that the leading UAV
                  reaches a point in LOS with the UE. 
                - if "nearest", the trajectory is planned so that the leading
                  UAV reaches the nearest grid point to the UE. 
                - if "min_ue_rate", the destination is any configuration such
                  that (i) there is LOS at all links AND (ii) the UE rate is
                  greater than `min_ue_rate`. If no such a configuration point
                  exists OR exists but is not reachable, planning a trajectory
                  will return None.

            - 'mode_connect':
                - if "min_rate_only", a configuration point is in Qfree if
                    + capacity(BS,UAV1) >= 2 * min_uav_rate
                    + capacity(UAV1, UAV2) >= min_uav_rate
                - otherwise, a configuratoin point is in Qfree if
                    + there is LoS from BS -> UAV1 -> UAV2
                    + capacity(BS, UAV1) >= 2*min_uav_rate
                    + capacity(UAV1, UAV2) >= min_uav_rate
            
            - b_conf_pts_meet_min_ue_rate: if True, all drawn configuration
              points meet the min_ue_rate requirement.
            
            - ue_rate_below_target_penalty: defined in planning path to serve
              moving ue.
            
            - b_tentative: if True, returns lm_path_tentative in planning path
              to serve moving ue.

        """

        super().__init__(**kwargs)
        self._num_uavs = num_uavs
        self._num_nodes = num_nodes
        self._max_num_neighbors = max_num_neighbors
        if step_connectivity is not None:
            self._step_connectivity = step_connectivity
        else:
            self._step_connectivity = np.sqrt(3) * np.max(
                self._environment.fly_grid.spacing)
        self.min_uav_rate = min_uav_rate
        self.fly_height = fly_height
        self._destination = destination
        self.min_ue_rate = min_ue_rate

        self.des_coor = des_coor

        self.l_des_set_ind = None
        self.mode_connect = mode_connect
        self.mode_draw_conf_pt = mode_draw_conf_pt
        self.b_conf_pts_meet_min_ue_rate = b_conf_pts_meet_min_ue_rate
        self.ue_rate_below_target_penalty = ue_rate_below_target_penalty
        self.b_tentative = b_tentative

    # @property
    # def name(self):
    #     if self.b_tentative:
    #         return self.__class__.__name__ + "Tentative"
    #     else:
    #         return self.__class__.__name__

    # @property
    # def name_on_figs(self):
    #     if self.b_tentative:
    #         return self._name_on_figs + " - Tentative"
    #     else:
    #         return self._name_on_figs

    def plan_path_to_serve_static_ue(self, bs_loc, ue_loc, samp_int,
                                     max_uav_speed):
        """
        TODO: write this docstring.
        """

        def plan_path_coarse(bs_loc, ue_loc):

            def get_conf_pt_inds_with_los_to_ue(t_conf_pt, ue_loc):
                """
                    Assumption: the first UAV in t_conf_pt is always in LoS with the base station.

                    Returns:
                        l_des_ind: list of indices of configuration pionts whose last uav is in LoS with the UE

                """

                l_des_ind = []
                for ind_conf_pt in range(t_conf_pt.shape[0]):

                    if self.are_in_los(t_conf_pt[ind_conf_pt, -1], ue_loc):
                        l_des_ind.append(ind_conf_pt)

                return l_des_ind

            def get_conf_pt_inds_with_at_least_min_ue_rate(
                    t_conf_pt, bs_loc, ue_loc):
                """     

                    Returns:
                        l_des_ind: list of indices of configuration points for which the UE rate exceeds self._min_ue_rate.

                """

                l_des_ind = []
                for ind_conf_pt in range(t_conf_pt.shape[0]):
                    rate = PathPlanner.conf_pt_to_ue_rate(
                        self._channel,
                        t_conf_pt[ind_conf_pt],
                        self.min_uav_rate,
                        bs_loc=bs_loc,
                        ue_loc=ue_loc)
                    if rate > self.min_ue_rate:
                        l_des_ind.append(ind_conf_pt)

                return l_des_ind

            def find_nearest_conf_pt_ind(t_conf_pt, conf_pt):
                return np.argmin(
                    np.sum(np.sum((t_conf_pt - conf_pt[None, :, :])**2,
                                  axis=1),
                           axis=1))

            def get_conf_pts(bs_loc, ue_loc, mode="random"):
                """ Returns a num_conf_pts x 2 x 3 array. All the configuration points
                must be in Qfree"""

                def get_conf_pts_random():
                    l_conf_pt = []
                    env = self._environment

                    while len(l_conf_pt) < self._num_nodes:

                        m_conf_pt = np.array([[
                            np.random.uniform(low=env.slf.grid.min_x,
                                              high=env.slf.grid.max_x),
                            np.random.uniform(low=env.slf.grid.min_y,
                                              high=env.slf.grid.max_y),
                            np.random.uniform(low=env.slf.grid.min_z,
                                              high=env.slf.grid.max_z)
                        ] for _ in range(self._num_uavs)])

                        if self._is_in_Qfree(m_conf_pt):
                            l_conf_pt.append(m_conf_pt)

                    return np.array(l_conf_pt)

                def get_conf_pts_random_at_inter():

                    def find_road_intersections():
                        """Finds the coordinates of the intersections of the block limits and the x and y axes.

                        Returns a (num_inter_y_axis x num_inter_x_axis x 2) tensor contains the coordinates of the intersections.
                        """

                        def find_road_coor_1d(block_limits,
                                              grid_max,
                                              safety_margin=5):
                            """ Given block_limits in 1 dimension

                                Returns the coordinates ("block_limits") of the roads in that dimension

                                Caution: if safety_margin < 5, self.is_in_Qfree may provide incorrect results maybe because of the mismatch of the real locations and those in the slf grid
                            
                            """
                            v_safety = safety_margin * np.array([[-1, 1]])
                            v_coor = list(np.ravel(block_limits + v_safety))
                            v_coor.insert(0, 0)
                            v_coor.append(grid_max)

                            return np.array(v_coor)

                        env = self._environment

                        v_x_coor_inter = find_road_coor_1d(
                            env.block_limits_x, env.slf.grid.max_x)

                        v_y_coor_inter = find_road_coor_1d(
                            env.block_limits_y, env.slf.grid.max_y)

                        num_inter_x = v_x_coor_inter.shape[0]
                        num_inter_y = v_y_coor_inter.shape[0]

                        m_x_coor = np.tile(
                            v_x_coor_inter.reshape(1, num_inter_x),
                            (num_inter_y, 1))
                        m_y_coor = np.tile(
                            v_y_coor_inter.reshape(num_inter_y, 1),
                            (1, num_inter_x))

                        return np.stack((m_x_coor, m_y_coor), axis=-1)

                    l_conf_pts = []

                    env = self._environment

                    # Discretize the configuration space
                    t_inter_pts = find_road_intersections()

                    def draw_a_conf_pt_at_intersections(t_inter_pts):

                        num_loc_x = t_inter_pts.shape[1]
                        num_loc_y = t_inter_pts.shape[0]

                        m_available_pts = np.ones((num_loc_x, num_loc_y),
                                                  dtype=bool)

                        l_uav_ind = []

                        # Append the 1st UAV
                        ind_x = np.random.randint(0, num_loc_x)
                        ind_y = np.random.randint(0, num_loc_y)
                        m_available_pts[ind_x,
                                        ind_y] = False  # make as not available
                        l_uav_ind.append([ind_x, ind_y])

                        # Draw the indices of the coordinates of other UAVs
                        # Iteratively draw
                        l_choice = ['row', 'col']
                        while len(l_uav_ind) < self._num_uavs:

                            previous_uav = l_uav_ind[-1]

                            if random.choice(l_choice) == 'row':
                                # The next uav will have the same row as the previous one
                                if np.mod(previous_uav[0], 2) == 0:
                                    ind_x = previous_uav[
                                        0] + np.random.randint(0, 2)
                                else:
                                    ind_x = previous_uav[
                                        0] - np.random.randint(0, 2)

                                ind_y = np.random.randint(0, num_loc_y)
                            else:
                                # The next uav will have the same column as the previous one
                                ind_x = np.random.randint(0, num_loc_x)

                                if np.mod(previous_uav[1], 2) == 0:
                                    ind_y = previous_uav[
                                        1] + np.random.randint(0, 2)
                                else:
                                    ind_y = previous_uav[
                                        1] - np.random.randint(0, 2)

                            if m_available_pts[ind_x, ind_y] == True:
                                l_uav_ind.append([ind_x, ind_y])
                                m_available_pts[ind_x, ind_y] = False

                        # Calculate the coordinates from the indices
                        l_uavs = []
                        for ind_uav in range(self._num_uavs):
                            ind_x = l_uav_ind[ind_uav][0]
                            ind_y = l_uav_ind[ind_uav][1]
                            coor = list(t_inter_pts[ind_x, ind_y])
                            # Add a random height
                            coor.append(
                                np.random.uniform(low=env.slf.grid.min_z,
                                                  high=env.slf.grid.max_z))
                            l_uavs.append(coor)

                        return np.array(l_uavs)

                    while len(l_conf_pts) < self._num_nodes:

                        l_conf_pts.append(
                            draw_a_conf_pt_at_intersections(t_inter_pts))

                    return np.array(l_conf_pts)

                def get_conf_pts_on_grid(bs_loc):

                    fly_grid = self._environment.fly_grid
                    m_grid_pts = fly_grid.list_pts()
                    num_grid_pts = m_grid_pts.shape[0]

                    def draw_a_conf_pt_in_los(bs_loc):
                        """
                            TODO: fix. 
                            
                            Returns a configuration point whose 1st UAV is in LoS with bs_loc
                        
                        """

                        def draw_a_uav_ind_in_los(m_grid_pts, pt1_loc,
                                                  l_available):
                            """
                            Draw a point on the fly grid and in LoS with pt1_loc

                            Args:

                            l_available: List of length m_grid_pts.shape[0] where the
                            l_available[i] == True if there is not any UAV at grid point i. 

                            Returns:
                                pt2_ind: the index of that point and grid l_available[pt2_ind] =
                                False
                            """

                            pt2_ind = np.random.randint(m_grid_pts.shape[0])

                            while all(m_grid_pts[pt2_ind] == pt1_loc) or (
                                    l_available[pt2_ind] == False):
                                pt2_ind = np.random.randint(
                                    m_grid_pts.shape[0])

                            if self.are_in_los(pt1_loc, m_grid_pts[pt2_ind]):
                                l_available[pt2_ind] = False
                                return pt2_ind, l_available
                            else:
                                return draw_a_uav_ind_in_los(
                                    m_grid_pts, pt1_loc, l_available)

                        l_available = num_grid_pts * [True]
                        # draw the 1st uav
                        uav1_ind, l_available = draw_a_uav_ind_in_los(
                            m_grid_pts, bs_loc, l_available)
                        l_uav_ind = [uav1_ind]

                        for ind_uav in range(1, self._num_uavs):

                            pt2_ind, l_available = draw_a_uav_ind_in_los(
                                m_grid_pts, m_grid_pts[l_uav_ind[ind_uav - 1]],
                                l_available)
                            l_uav_ind.append(pt2_ind)

                        return m_grid_pts[l_uav_ind]

                    l_conf_pts = []

                    if self._num_uavs == 1:

                        for ind in range(num_grid_pts):
                            if self.are_in_los(m_grid_pts[ind], bs_loc):
                                l_conf_pts.append(m_grid_pts[None, ind])
                        self._num_nodes = len(l_conf_pts)

                    else:
                        while len(l_conf_pts) < self._num_nodes:

                            # uav1 of every conf_pt should always be in LoS with the BS
                            conf_pt = draw_a_conf_pt_in_los(bs_loc)

                            l_conf_pts.append(conf_pt)

                            # drop conf_pt if it is already in l_conf_pts
                            if len(l_conf_pts) >= 2:
                                for ind in range(len(l_conf_pts) - 1):
                                    if np.linalg.norm(
                                            np.ravel(conf_pt -
                                                     l_conf_pts[ind])) == 0:
                                        l_conf_pts.pop()
                                        break

                    return np.array(l_conf_pts) + np.random.rand(
                        len(l_conf_pts), self._num_uavs, 3)

                def get_conf_pts_from_core(bs_loc, ue_loc):
                    """
                    This function produces a set of configuration points for 2 UAVs where:

                    The locations of UAV2 are drawn uniformly at random from the shortest
                    path between bs_loc and the destination. The shortest path goes through
                    points that have LoS with at least one point that has LoS with the BS. 

                    The locations of UAV1 are drawn from the locations with LOS with the BS
                    with a probabability that is proportional to the number of points in the
                    aforementioned shortest path that are in LOS with that location.
                    
                    """

                    def get_pts_in_los_with_bs(bs_loc):
                        fly_grid = self._environment.fly_grid
                        m_grid_pts = fly_grid.list_pts()
                        l_pts_in_los_with_bs = [
                            m_grid_pts[ind] for ind in
                            self._get_grid_pt_inds_in_los_with(bs_loc)
                        ]

                        return l_pts_in_los_with_bs

                    l_pts_in_los_with_bs = get_pts_in_los_with_bs(bs_loc)

                    assert len(l_pts_in_los_with_bs) != 0

                    # Plan trajectory of UAV2 along points in LOS with at least a point in `l_pts_in_los_with_bs`
                    m_shortest_path_uav_2 = np.array(
                        self.
                        _get_shortest_path_uav2_thru_pts_with_losNrcc_with_pts_with_losN2rcc_with_bs(
                            bs_loc, ue_loc))[:, 0, :]

                    # For each pt in l_pts_in_los_with_bs, count how many points in
                    # l_shortest_path_uav_2 are in LOS. This can be used to obtain a
                    # sampling distribution to draw locations of UAV2 from
                    # `l_pts_in_los_with_bs`

                    # For point n in l_pts_in_los_with_bs, count the number of points in m_shortest_path_uav_v that are in LoS with point n.
                    v_num_pts_in_los = np.zeros((len(l_pts_in_los_with_bs), ))

                    # For each position of uav_2, store the inds of the uav_1 positions in LOS
                    num_possible_qpts = 0
                    l_inds_uav_1_in_los_with_uav_2 = [
                        [] for _ in range(len(m_shortest_path_uav_2))
                    ]
                    for ind_pt_uav_2 in range(len(m_shortest_path_uav_2)):
                        pt_uav_2 = m_shortest_path_uav_2[ind_pt_uav_2]
                        for ind_pt_uav_1 in range(len(l_pts_in_los_with_bs)):
                            pt_uav_1 = l_pts_in_los_with_bs[ind_pt_uav_1]
                            if self.are_in_los(pt_uav_1, pt_uav_2):
                                v_num_pts_in_los[ind_pt_uav_1] += 1
                                l_inds_uav_1_in_los_with_uav_2[
                                    ind_pt_uav_2].append(ind_pt_uav_1)
                                num_possible_qpts += 1

                    # Draw conf pts
                    l_inds_locs_uav_1 = []
                    l_inds_locs_uav_2 = []
                    v_num_remaining_uav_1_pts = np.array(
                        [len(l) for l in l_inds_uav_1_in_los_with_uav_2])

                    for _ in range(
                            np.minimum(self._num_nodes, num_possible_qpts)):

                        ind_loc_uav_2 = np.random.choice(
                            np.where(v_num_remaining_uav_1_pts > 0)[0])
                        l_inds_locs_uav_2.append(ind_loc_uav_2)

                        # Get indices of locs in LoS with (ind_loc_uav_2 + bs)
                        v_inds_los_locs_uav_1 = np.array(
                            l_inds_uav_1_in_los_with_uav_2[ind_loc_uav_2])

                        # The number of points in m_shortest_path_uav_v that are in LoS
                        # each point in v_inds_los_locs_uav_1
                        v_prob = v_num_pts_in_los[v_inds_los_locs_uav_1]
                        v_prob = v_prob / np.sum(v_prob)

                        ind_loc_uav_1 = np.random.choice(v_inds_los_locs_uav_1,
                                                         p=v_prob)
                        l_inds_locs_uav_1.append(ind_loc_uav_1)

                        l_inds_uav_1_in_los_with_uav_2[ind_loc_uav_2].remove(
                            ind_loc_uav_1)
                        v_num_remaining_uav_1_pts[ind_loc_uav_2] -= 1

                    m_locs_uav_2 = np.array(
                        m_shortest_path_uav_2)[l_inds_locs_uav_2]
                    m_locs_uav_1 = np.array(
                        l_pts_in_los_with_bs)[l_inds_locs_uav_1]

                    t_conf_pts = np.concatenate(
                        (m_locs_uav_1[:, None, :], m_locs_uav_2[:, None, :]),
                        axis=1)

                    return np.array(t_conf_pts)

                def get_conf_pts_from_feasible_path(bs_loc, ue_loc):

                    def get_feasible_trajectory(bs_loc, ue_loc):
                        """"
                            Returns: list of 2 x 3 conf. pts with a feasible trajectory for both
                            UAVs if it exists. If it does not exist, it returns None.        
                        """

                        # UAV 2
                        if self.mode_connect == 'min_rate_only':
                            l_path_uav2 = self._get_shortest_path_uav_2_thru_pts_greater_rate_with_pts_greater_rate_with_bs(
                                bs_loc, ue_loc)
                        else:
                            l_path_uav2 = self._get_shortest_path_uav2_thru_pts_with_losNrcc_with_pts_with_losN2rcc_with_bs(
                                bs_loc, ue_loc)

                        if l_path_uav2 is None:
                            # This happens when self.min_ue_rate is too high.
                            return None

                        lm_path = self._get_path_both_uavs_given_path_uav2_2serve_static_ue(
                            bs_loc, ue_loc, l_path_uav2)

                        return lm_path

                    lm_feas_path = get_feasible_trajectory(bs_loc, ue_loc)
                    if lm_feas_path is None:
                        return None
                    else:

                        # Add the initial locations to make sure that it is
                        # possible to go from the bs to the first configuration
                        # point.
                        m_uav_loc_start = np.tile(bs_loc, (self._num_uavs, 1))
                        lm_feas_path = [m_uav_loc_start] + lm_feas_path

                    if self.b_tentative:
                        return lm_feas_path

                    t_all_pts = self._environment.fly_grid.list_pts()
                    l_feas_pt_inds_uav1 = self._get_grid_pt_inds_in_los_n_minrate_with(
                        bs_loc)
                    l_feas_pts_uav1 = [
                        t_all_pts[ind] for ind in l_feas_pt_inds_uav1
                    ]
                    l_feas_pt_inds_uav2 = [
                        ind for ind in range(len(t_all_pts))
                        if self.are_in_los(t_all_pts[ind], l_feas_pts_uav1)
                    ]
                    l_feas_pts_uav2 = [
                        t_all_pts[ind] for ind in l_feas_pt_inds_uav2
                    ]

                    return np.array(
                        self._sample_around_tentative_path(
                            lm_feas_path, l_feas_pts_uav1, l_feas_pts_uav2))

                if mode == "random":
                    return get_conf_pts_random()
                elif mode == "inter":
                    return get_conf_pts_random_at_inter()
                elif mode == "grid":
                    return get_conf_pts_on_grid(bs_loc)
                elif mode == "core":
                    return get_conf_pts_from_core(bs_loc, ue_loc)
                elif mode == "feasible":
                    return get_conf_pts_from_feasible_path(bs_loc, ue_loc)

            self.bs_loc = bs_loc

            # Draw configuration points
            t_conf_pt = get_conf_pts(bs_loc,
                                     ue_loc,
                                     mode=self.mode_draw_conf_pt)
            if t_conf_pt is None:

                print('RR: t_conf_pt is None, return None')
                return None

            if self.b_tentative:
                print("                . Return tentative path")
                return t_conf_pt

            # Compute the cost
            m_cost = self._get_cost_all_cf_pts_2serve_static_ue(t_conf_pt)

            # Get destination configuration pionts whose last uav is in LoS with the UE
            if self._destination == "los":
                des_ind = get_conf_pt_inds_with_los_to_ue(t_conf_pt, ue_loc)
            elif self._destination == "nearest":
                des_ind = find_nearest_conf_pt_ind(
                    t_conf_pt[:, -1, :][:, None, :], ue_loc[None, :])
            elif self._destination == "min_ue_rate":
                des_ind = get_conf_pt_inds_with_at_least_min_ue_rate(
                    t_conf_pt, bs_loc, ue_loc)
            else:
                raise ValueError

            l_shortest_path_inds = self.get_shortest_path(
                m_cost, ind_node_start=0, ind_nodes_end=des_ind)

            if l_shortest_path_inds is not None:
                l_shortest_path = [
                    t_conf_pt[ind] for ind in l_shortest_path_inds
                ]
                return l_shortest_path
            else:
                # This may happen if there are corners that make that adjacent grid
                # points are found not to be connected by self.get_all_costs().

                print('RR cannot find the shortest path returns None')
                return None

        print(f'    o {self.name}')

        self.bs_loc = bs_loc
        lm_path_coarse = plan_path_coarse(bs_loc, ue_loc)
        lm_path = self.resample(lm_path_coarse, samp_int, max_uav_speed)
        return lm_path

    def plan_path_to_serve_moving_ue(self, bs_loc, lm_ue_path, debug=False):
        """
        Plan paths for the UAVs to maximize the cumulative rate of the UE over
        its trajectory.

        Args: 
            + lm_ue_path: list of (1,3) matrices of UE locations. The n-th
            matrix contains the UE locations at time step n.

            + self.ue_rate_below_target_penalty: cost of not being able to
              provide the min_ue_rate to the UE in the next time step of UAV-1
              and UAV-2. This is used to compute the tentativ path.

        Reuturns:
            
            + lm_path: list of (2,3) matrices of the locations of UAV-1 and
              UAV-2. The n-th matrix contains the locations of UAV-1 and UAV-2
              at time step n.

              If self.b_tentative, lm_path contains the tentative path. Else, it
              contains the path obtained by probabilistic roadmaps. 

        """

        print(f'    o {self.name}')

        self.bs_loc = bs_loc

        m_uav_loc_start = np.tile(bs_loc, (self._num_uavs, 1))
        # Drop the first location of the UE path since UAVs start at
        # m_uav_loc_start
        lm_ue_path_trim = lm_ue_path[1:]

        def plan_tentative_path(bs_loc, lm_ue_path):
            """
            UAV-2 flies in the set of feasible locations R(qBS, 2rcc, rcc) which
            includes its destinations at every time step R(qBS, 2rcc+r_min_ue,
            rcc+r_min_ue) \cap R(qUE[n],r_min_ue). From time step n to (n +1),
            traveling to a location that is not a destination of time step (n+1)
            has higher cost than traveling to a destination.
            
            At time step n, there is a set of grid points where UAV-1 has at
            least 2rcc with the BS and rcc with UAV-2. That set includes the
            locations where UAV-1 has at least 2rcc + r_min_ue with the BS and
            rcc + r_min_ue with UAV-2. Such locations are called as the
            destinations of UAV-1. Similar to UAV-2, from time step n to n+1,
            traveling to a destination of time instant (n+1) has lower cost
            than traveling to a location that is not a destination.    
            """

            def einds_to_grid_pts(fly_grid, l_path_einds, ll_dest_inds):
                l_path_fly_grid_pt_inds = [
                    self._eind_to_block_and_fly_grid_pt_ind(
                        eind, ll_dest_inds)[1] for eind in l_path_einds
                ]

                m_grid_pts = fly_grid.list_pts()
                lm_path = [
                    m_grid_pts[grid_pt_ind, None, :]
                    for grid_pt_ind in l_path_fly_grid_pt_inds
                ]

                return lm_path

            def get_feas_and_dest_uav1_wrt_bs(bs_loc):
                """                
                Find feasible locations and destinations of UAV-1 given the BS
                location.

                    + Feasible locations: where UAV-1 has at least
                      2*min_uav_rate with the BS.

                    + Destination: where UAV-1 has at least 2*min_uav_rate +
                      min_ue_rate with the BS.

                Note: wrt_bs means regardless of UAV-2. 

                Returns:

                - fly_grid_feas: grid of feasible locations of UAV-1 wrt the BS.
                """

                # Feasible locations
                fly_grid_feas = self._environment.fly_grid.clone()
                fly_grid_feas.disable_by_indicator(
                    lambda pt: not self.is_at_least_rate(
                        pt, [bs_loc], 2 * self.min_uav_rate))
                m_grid_pts = fly_grid_feas.list_pts()
                l_feas_inds_wrt_bs = list(range(len(m_grid_pts)))
                l_feas_wrt_bs = list(m_grid_pts)

                # Destinations
                l_dest_inds_wrt_bs = self._get_grid_pt_inds_in_minrate_with(
                    [bs_loc],
                    2 * self.min_uav_rate + self.min_ue_rate,
                    fly_grid=fly_grid_feas)
                l_dest_wrt_bs = list(m_grid_pts[l_dest_inds_wrt_bs])

                return fly_grid_feas, l_feas_inds_wrt_bs, l_feas_wrt_bs, l_dest_inds_wrt_bs, l_dest_wrt_bs

            def get_feas_and_dest_uav2(lm_ue_path, l_feas_uav1_wrt_bs,
                                       l_dest_uav1_wrt_bs):
                """
                Find feasible locations and destination of UAV-2.
                
                    + Feasible locations: where UAV-2 has at least min_uav_rate
                      with at least a point that has at least 2*min_uav_rate
                      with the BS.
                    
                    + Destinations: where UAV-2 has at least min_uav_rate +
                      min_ue_rate with at least a point that has at least
                      2*min_uav_rate + min_ue_rate with the BS and at least
                      min_ue_rate with the UE. The set of destinations changes
                      as the UE moves. 
                """
                fly_grid_feas = self._environment.fly_grid.clone()
                fly_grid_feas.disable_by_indicator(
                    lambda pt: not self.is_at_least_rate(
                        pt, l_feas_uav1_wrt_bs, self.min_uav_rate))

                # --- find the set of inds of destinations of UAV-2 wrt UAV-1
                s_dests_inds_wrt_uav1 = set(
                    self._get_grid_pt_inds_in_minrate_with(
                        l_dest_uav1_wrt_bs,
                        self.min_uav_rate + self.min_ue_rate,
                        fly_grid=fly_grid_feas))

                # --- find the list of lists of inds of destinations of UAV-2 at each time instant
                ll_dest_inds = []
                for m_loc in lm_ue_path:
                    l_dest_inds_wrt_ue = self._get_grid_pt_inds_in_minrate_with(
                        m_loc[0], self.min_ue_rate, fly_grid=fly_grid_feas)
                    l_dest_inds = list(
                        s_dests_inds_wrt_uav1.intersection(l_dest_inds_wrt_ue))
                    ll_dest_inds.append(l_dest_inds)

                return fly_grid_feas, ll_dest_inds

            def get_m_cost_extended_graph_weighted(fly_grid,
                                                   ll_uav_des_inds_vs_time,
                                                   ll_feas_inds=None):
                """
                Args:

                    - If ll_feas_inds is None, the output matrix contains the
                      cost between each pair of (block, grid point). 


                Returns:
                    + m_costs is a matrix of size (num_ex_nodes, num_ex_nodes)
                      where num_ex_nodes is

                        . the total number of entries in ll_loc_inds (if not
                        None), or

                        . num_grid_pts * len(ll_uav_des_inds_vs_time).
                        
                    m_cost[i,j] entry is the cost of going from ex_node i to
                    ex_node j.                            

                    C1. the grid points corresponding to ex_nodes i and j are
                        adjacent.
                            
                    C2. the block corresponding to ex_node j is the next
                        block to the block corresponding to ex_node i. 

                    C3. the grid point corresponding to ex_node j is in
                        ll_uav_des_inds_vs_time[n], where n is the index of the
                        block corresponding to ex_node j.
                    
                    C4. the grid point corresponding to ex_node i is the grid
                        point corresponding to ex_node j. 

                    m_costs[i,j] is:
                    
                        - 0 if (C1, C2, C3, C4) are true. The grid point
                          corresponding to ex_node i is the gridpoint
                          corresponding to ex_node j and the gridpoint
                          corresponding to node j is a destination. 

                        - 1 if (C1, C2, C3) are true and C4 false. (traveling to
                          a destination in the next time step. The destination
                          is adjacent to the current location of UAV-2)

                        - self.ue_rate_below_target_penalty if C1 and C2 are
                          true but C3 is false. (traveling to not a destination
                          in the next time step)

                        - Inf otherwise. (cannot go to a grid point that is not
                          adjacent to the current grid point or cannot go to the
                          block that is not the next block to the current block)
                """

                # Total number of nodes
                num_blocks = len(ll_uav_des_inds_vs_time)
                m_grid_pts = fly_grid.list_pts()

                num_grid_pts = m_grid_pts.shape[0]
                if ll_feas_inds is None:
                    num_ex_nodes = num_grid_pts * num_blocks
                else:
                    num_ex_nodes = 0
                    for l_ind_loc in ll_feas_inds:
                        num_ex_nodes += len(l_ind_loc)
                if debug:
                    print(
                        '------ Warning: if num_ex_nodes is too large, the algorithm may stop without returning an error, which may be due to running out of memory.'
                    )
                m_cost = np.full((num_ex_nodes, num_ex_nodes), np.inf)

                # Compute a num_grid_pts x num_grid_pts matrix whose [i,j]-th
                # entry is ue_rate_below_target_penalty if C1 and np.Inf
                # otherwise.
                m_block_base = np.full((num_grid_pts, num_grid_pts), np.Inf)
                for ind_1 in range(num_grid_pts):
                    for ind_2 in range(ind_1, num_grid_pts):
                        if ind_1 == ind_2:
                            m_block_base[
                                ind_1,
                                ind_2] = self.ue_rate_below_target_penalty
                            continue

                        if fly_grid.are_adjacent(m_grid_pts[ind_1],
                                                 m_grid_pts[ind_2]):
                            if self.are_in_los(m_grid_pts[ind_1],
                                               m_grid_pts[ind_2]):
                                m_block_base[
                                    ind_1,
                                    ind_2] = self.ue_rate_below_target_penalty
                                m_block_base[ind_2,
                                             ind_1] = m_block_base[ind_1,
                                                                   ind_2]

                # For each block, copy m_block_base and set the necessary entries to 0 or 1.
                def get_block(ind_block):

                    l_inds_dest = ll_uav_des_inds_vs_time[ind_block + 1]
                    m_block = np.copy(m_block_base)

                    # (C1, C2, C3) are true
                    m_block[:,
                            l_inds_dest] = m_block[:,
                                                   l_inds_dest] / self.ue_rate_below_target_penalty
                    # (C1, C2, C3, C4) are true
                    for ind in l_inds_dest:
                        m_block[ind, ind] = 0

                    if ll_feas_inds is not None:
                        m_block = m_block[
                            ll_feas_inds[ind_block], :][:,
                                                        ll_feas_inds[ind_block
                                                                     + 1]]

                    return m_block

                ind_block_start_row = 0

                for ind_block in range(num_blocks - 1):

                    num_block_rows = len(
                        ll_feas_inds[ind_block]
                    ) if ll_feas_inds is not None else num_grid_pts

                    num_block_cols = len(ll_feas_inds[
                        ind_block +
                        1]) if ll_feas_inds is not None else num_grid_pts

                    ind_block_end_row = ind_block_start_row + num_block_rows
                    ind_block_end_next_col = ind_block_end_row + num_block_cols

                    # C2
                    m_cost[
                        ind_block_start_row:ind_block_end_row,
                        ind_block_end_row:ind_block_end_next_col] = get_block(
                            ind_block)

                    ind_block_start_row = ind_block_end_row

                return m_cost

            def get_path_uav2(fly_grid, ll_ind_des):
                # --- compute costs
                m_cost = get_m_cost_extended_graph_weighted(
                    fly_grid, ll_ind_des)

                # --- find the shortest path
                num_grid_pts = fly_grid.list_pts().shape[0]
                ll_loc_inds = [list(range(num_grid_pts))] * len(ll_ind_des)

                enode_ind_start = self._block_and_fly_grid_pt_inds_to_einds(
                    0, fly_grid.nearest_ind(bs_loc), ll_loc_inds)[0]

                l_enode_inds_end = self._block_and_fly_grid_pt_inds_to_einds(
                    len(ll_ind_des) - 1, ll_loc_inds[-1],
                    ll_loc_inds)  # ll_ind_des[-1]

                t0 = time.time()
                l_path_einds_uav = self.get_shortest_path(
                    m_cost,
                    ind_node_start=enode_ind_start,
                    ind_nodes_end=l_enode_inds_end,
                    num_nodes_per_block=num_grid_pts)
                if debug:
                    print(
                        f'-- Planed tentative path for UAV-2, time: {time.time() - t0} s'
                    )

                # --- einds to block and grid point indices
                lm_path_uav = einds_to_grid_pts(fly_grid, l_path_einds_uav,
                                                ll_loc_inds)

                return lm_path_uav

            def plan_path_both_uavs_given_uav2_path_moving_ue(
                    fly_grid_uav1, l_feas_inds_uav1_wrt_bs,
                    l_dest_inds_uav1_wrt_bs, lm_path_uav2):

                def get_locs_n_dests_uav1(fly_grid, l_feas_inds_wrt_bs,
                                          l_des_inds_wrt_bs, lm_path_uav2):
                    """	
                    Find the feasible locations and destinations of UAV-1 wrt
                    UAV-2 for each time instant of lm_path_uav2.
                    """

                    ll_loc_inds = []
                    ll_dest_inds = []
                    for m_loc in lm_path_uav2:
                        # feasible locations for UAV1
                        l_feas_inds_wrt_uav2 = self._get_grid_pt_inds_in_minrate_with(
                            m_loc[0], self.min_uav_rate, fly_grid=fly_grid)
                        ll_loc_inds.append(
                            list(
                                set(l_feas_inds_wrt_uav2).intersection(
                                    l_feas_inds_wrt_bs)))

                        # destinations
                        l_dest_inds_wrt_uav2 = self._get_grid_pt_inds_in_minrate_with(
                            m_loc[0],
                            self.min_uav_rate + self.min_ue_rate,
                            fly_grid=fly_grid)
                        l_dest_inds = list(
                            set(l_des_inds_wrt_bs).intersection(
                                l_dest_inds_wrt_uav2))
                        ll_dest_inds.append(l_dest_inds)

                    return ll_loc_inds, ll_dest_inds

                def get_path_uav1(fly_grid, ll_dest_inds, ll_feas_inds=None):

                    # --- compute costs
                    m_cost = get_m_cost_extended_graph_weighted(
                        fly_grid, ll_dest_inds, ll_feas_inds=ll_feas_inds)

                    # --- find the shortest path
                    enode_ind_start = self._block_and_fly_grid_pt_inds_to_einds(
                        0, fly_grid.nearest_ind(self.bs_loc), ll_feas_inds)[0]
                    l_enode_inds_end = self._block_and_fly_grid_pt_inds_to_einds(
                        len(ll_dest_inds) - 1, ll_feas_inds[-1],
                        ll_feas_inds)  # ll_dest_inds[-1]
                    l_num_nodes_per_block = [
                        len(l_ind_loc) for l_ind_loc in ll_feas_inds
                    ]
                    t0 = time.time()
                    l_path_einds_uav = self.get_shortest_path(
                        m_cost,
                        ind_node_start=enode_ind_start,
                        ind_nodes_end=l_enode_inds_end,
                        num_nodes_per_block=l_num_nodes_per_block)

                    if l_path_einds_uav is None:
                        return None
                    if debug:
                        print(
                            f'-- Planed path for UAV-1, time: {time.time() - t0} s'
                        )

                    # --- einds to block and grid point indices
                    lm_path_uav = einds_to_grid_pts(fly_grid, l_path_einds_uav,
                                                    ll_feas_inds)
                    return lm_path_uav

                initial_pos_uav_2, final_pos_uav_2 = lm_path_uav2[
                    0], lm_path_uav2[-1]

                while True:
                    self._concat_takeoff_and_landing(lm_path_uav2,
                                                     initial_pos_uav_2,
                                                     final_pos_uav_2)
                    ll_feas_inds_uav1, ll_des_ind_uav1 = get_locs_n_dests_uav1(
                        fly_grid_uav1, l_feas_inds_uav1_wrt_bs,
                        l_dest_inds_uav1_wrt_bs, lm_path_uav2)

                    lm_path_uav1 = get_path_uav1(
                        fly_grid_uav1,
                        ll_des_ind_uav1,
                        ll_feas_inds=ll_feas_inds_uav1)

                    if lm_path_uav1 is not None:
                        break

                    lm_path_uav2 = self._lift_path(lm_path_uav2)

                    if lm_path_uav2 is None:
                        raise ValueError(
                            "No path can be found even by lifting the path of UAV2. Most likely some buildings are higher than the highest grid point. "
                        )

                assert lm_path_uav1 is not None, 'Cannot find a path for UAV-1'

                # concatenate the paths of the uavs
                lm_path = []
                for ind in range(len(lm_path_uav2)):
                    lm_path.append(
                        np.concatenate((lm_path_uav1[ind], lm_path_uav2[ind]),
                                       axis=0))

                # find the list of destinations of UAV-1
                l_dests_uav1 = list(fly_grid_uav1.list_pts()[list(
                    set().union(*ll_des_ind_uav1))])

                return lm_path, l_dests_uav1

            # Find feasible locations and destinations of UAV-1 wrt the BS
            # (regardless of UAV-2)
            fly_grid_uav1, l_feas_inds_uav1_wrt_bs, l_feas_uav1_wrt_bs, \
                l_dest_inds_uav1_wrt_bs, l_dest_uav1_wrt_bs = get_feas_and_dest_uav1_wrt_bs(
                bs_loc)
            if len(l_dest_inds_uav1_wrt_bs) == 0:
                print(
                    f'[{self.name}] No feasible locations for UAV-1 to guarantee the min ue rate'
                )
                return None, None, None

            # Find feasible locations and destinations of UAV-2
            fly_grid_uav2, ll_dest_inds_uav2 = get_feas_and_dest_uav2(
                lm_ue_path, l_feas_uav1_wrt_bs, l_dest_uav1_wrt_bs)
            # Find the list of destinations of UAV-2
            l_dests_uav2 = list(fly_grid_uav2.list_pts()[list(
                set().union(*ll_dest_inds_uav2))])

            if debug:
                print("--- Planning path for UAV-2")
            lm_path_uav2 = get_path_uav2(fly_grid_uav2, ll_dest_inds_uav2)

            if debug:
                print("\n--- Planning path for UAV-1")
            lm_path, l_dests_uav1 = plan_path_both_uavs_given_uav2_path_moving_ue(
                fly_grid_uav1, l_feas_inds_uav1_wrt_bs,
                l_dest_inds_uav1_wrt_bs, lm_path_uav2)

            return lm_path, l_dests_uav1, l_dests_uav2

        def plan_path_in_conf_space(lm_ue_path, llm_conf_pts):

            # construct m_cost for l_conf_pts
            def get_m_cost_extended_graph_conf_pts(llm_conf_pts,
                                                   lm_ue_path=None):
                """

                Args:
                    - llm_conf_pts: list of lists of (2,3) matrices.
                      llm_conf_pts[n] is the list of conf pts at time step n.

                    - lm_ue_path: list of (1,3) matrices of UE locations. It
                      cannot be None.              

                Returns: 
                    - m_cost: a matrix of size (num_ex_nodes, num_ex_nodes)
                      where num_ex_nodes is the sum of the lengths of the lists
                      in llm_conf_pts.
                     
                Let 
                    - conf_pt_i be the configuration point corresponding to
                      ex_node i. 
                    
                    - conf_pt_j be the configuration point corresponding to
                      ex_node j.

                m_cost[i,j] is:
                    - -rate_ue provided by conf_pt_j if the following conditions
                      are true:
                        . the block corresponding to ex_node j is the next block
                        to the block corresponding to ex_node i.

                        . the grid points of conf_pt_i are respectively adjacent
                        to those of conf_pt_j.

                    - np.Inf otherwise.
                """

                assert lm_ue_path is not None

                def get_block(ind_block):

                    num_pts_curr_block = len(llm_conf_pts[ind_block])
                    num_pts_next_block = len(llm_conf_pts[ind_block + 1])
                    m_block = np.full((num_pts_curr_block, num_pts_next_block),
                                      np.inf)

                    for ind_1 in range(num_pts_curr_block):
                        for ind_2 in range(num_pts_next_block):
                            # need not to check for the connection between the
                            # conf. pts because the UAVs only move to their
                            # adjacent grid points, implemented in
                            # self._get_cost_btw_conf_pts()
                            m_block[ind_1,
                                    ind_2] = self._get_cost_btw_conf_pts(
                                        llm_conf_pts[ind_block][ind_1],
                                        llm_conf_pts[ind_block + 1][ind_2],
                                        loc_ue=lm_ue_path[ind_block + 1][0])

                            if ind_1 == 0 and ind_2 == 0 and m_block[
                                    ind_1, ind_2] == np.inf:
                                raise ValueError()

                    return m_block

                num_blocks = len(llm_conf_pts)

                num_ex_nodes = 0
                for lm_conf_pts in llm_conf_pts:
                    num_ex_nodes += len(lm_conf_pts)
                m_cost = np.full((num_ex_nodes, num_ex_nodes), np.inf)

                ind_block_start_row = 0

                for ind_block in range(num_blocks - 1):
                    ind_block_end_row = ind_block_start_row + len(
                        llm_conf_pts[ind_block])
                    ind_block_end_next_col = ind_block_end_row + len(
                        llm_conf_pts[ind_block + 1])
                    # block diagonal right above the main block diagonal
                    m_cost[
                        ind_block_start_row:ind_block_end_row,
                        ind_block_end_row:ind_block_end_next_col] = get_block(
                            ind_block)

                    ind_block_start_row = ind_block_end_row

                return m_cost

            m_cost = get_m_cost_extended_graph_conf_pts(llm_conf_pts,
                                                        lm_ue_path=lm_ue_path)

            # plan path through m_cost
            l_num_conf_pts_per_block = [
                len(lm_conf_pts) for lm_conf_pts in llm_conf_pts
            ]
            # find the first index of the last block
            v_eind_first_enode_each_block = self.get_eind_first_enode_each_block(
                llm_conf_pts)
            l_des_inds = list(
                np.arange(v_eind_first_enode_each_block[-2],
                          v_eind_first_enode_each_block[-1]))
            l_einds_path_min_outage = self.get_shortest_path(
                m_cost,
                ind_node_start=0,
                ind_nodes_end=l_des_inds,
                num_nodes_per_block=l_num_conf_pts_per_block)

            # from einds to block and grid point indices
            lm_path_min_outage = []
            for eind in l_einds_path_min_outage:
                _, conf_pts = self._eind_to_block_and_fly_grid_pt_ind(
                    eind, llm_conf_pts)
                lm_path_min_outage += [conf_pts]

            return lm_path_min_outage

        print("        . Planning tentative path")
        lm_path_tentative, l_dests_uav1, l_dests_uav2 = plan_tentative_path(
            bs_loc, lm_ue_path_trim)

        if lm_path_tentative is None:
            print('        . Cannot find a tentative path, return None')
            return None

        if self.b_tentative or len(l_dests_uav2) == 0:
            # this happens when self.min_ue_rate is too high.
            print("                . Return tentative path")
            return [m_uav_loc_start] + lm_path_tentative

        print(
            "        . Sampling configuration points around the tentative path"
        )
        llm_conf_pts = self._sample_around_tentative_path(
            lm_path_tentative,
            l_dests_uav1,
            l_dests_uav2,
            lm_ue_path=lm_ue_path_trim)

        print("        . Planning path in C-space")
        lm_path = plan_path_in_conf_space(lm_ue_path_trim, llm_conf_pts)

        return [m_uav_loc_start] + lm_path

    def _get_grid_pt_inds_in_los_with(self, ref_pts, fly_grid=None):
        """
        Args:

        ref_pt is either a 3D point or a list/np.array of 3D points.
        
        Returns:

        list of indices of the fly grid points that are in LOS with at least one
        point in `ref_pts`.
        
        """

        if fly_grid is None:
            fly_grid = self._environment.fly_grid.clone()

        fvf = FunctionVectorField(
            grid=fly_grid,
            fun=lambda pt: np.array([self.are_in_los(pt, ref_pts)]))

        return list(np.where(fvf.list_vals()[:, 0])[0])

    def _get_grid_pt_inds_in_minrate_with(self,
                                          ref_pts,
                                          min_rate,
                                          fly_grid=None):
        """
        Args:

        ref_pts is either a 3D point or a list/np.array of 3D points.
        
        Returns:

        list of indices of the fly grid points that have at least min_rate with
        at least one point in `ref_pts`.
        
        """

        if fly_grid is None:
            fly_grid = self._environment.fly_grid.clone()

        fvf = FunctionVectorField(
            grid=fly_grid,
            fun=lambda pt: np.array(
                [self.is_at_least_rate(pt, ref_pts, min_rate)]))

        return list(np.where(fvf.list_vals()[:, 0])[0])

    def _get_grid_pt_inds_in_los_n_minrate_with(self, bs_loc, min_rate=None):

        l_grid_pt_inds_in_los = self._get_grid_pt_inds_in_los_with(bs_loc)

        if min_rate is None:
            min_rate = 2 * self.min_uav_rate
        l_grid_pt_inds_greater_rate = self._get_grid_pt_inds_in_minrate_with(
            [bs_loc], min_rate)

        l_grid_pt_inds = list(
            set(l_grid_pt_inds_greater_rate).intersection(
                l_grid_pt_inds_in_los))

        return l_grid_pt_inds

    def _get_cost_all_cf_pts_2serve_static_ue(self,
                                              t_conf_pt,
                                              grid=None,
                                              neighbor_mode='configuration'):
        """ 
        Args: 
            `t_conf_pt`: num_nodes x num_uavs x 3. All nodes must be in Q-free.

            `neighbor_mode`: can be:

                - 'configuration': to determine the conf pts connected to
                conf_pt, its connectivity to all other conf pts is tested
                until self._max_num_neighbors neighbors are found, if possible. 

                - 'grid': only for num_uavs = 1. m_conf_pt_1 is a neighbor of
                m_conf_pt_2 if both points are adjacent on the grid.

        Returns:
            `m_cost`: num_nodes x num_nodes matrix whose (m,n)-th
        entry is
                - np.Inf if node m and node n are not connected          
                - the cost of going from node n to node m if nodes m and n are
                connected. Nodes m and n are connected iff:
                    + either m is within the nearest neighbors of n or n is
                    within the nearest neighbors of m, AND
                    + self.are_connected(t_conf_pt[m], t_conf_pt[n]) == True.
                
        """

        if grid is None:
            grid = self._environment.fly_grid

        def get_inds_neighbors(m_conf_pt, t_conf_pts):
            if neighbor_mode == 'configuration':
                """Returns a vector with the indices of the neighbors of
                m_conf_pt sorted in increasing order of distance. """
                v_dist = np.max(np.sum((t_conf_pts - m_conf_pt)**2, axis=2),
                                axis=1)
                return np.argsort(v_dist)
            elif neighbor_mode == 'grid':
                assert m_conf_pt.shape[0] == 1
                assert t_conf_pts.shape[1] == 1

                return self._environment.fly_grid.get_inds_adjacent(
                    m_conf_pt[0], t_conf_pts[:, 0, :])

        num_nodes = t_conf_pt.shape[0]

        # Determine if each conf pt is connected to its nearest neighbors
        m_cost = np.tile(np.inf, (num_nodes, num_nodes))

        for ind_conf_pt, v_conf_pt in enumerate(t_conf_pt):
            for ind_nn in get_inds_neighbors(v_conf_pt, t_conf_pt):
                if np.sum(m_cost[ind_conf_pt, :] <
                          np.inf) >= self._max_num_neighbors:
                    break

                if m_cost[ind_conf_pt, ind_nn] < np.inf:
                    continue

                if self._are_connected(v_conf_pt, t_conf_pt[ind_nn]):
                    m_cost[ind_conf_pt, ind_nn] = self._get_cost_btw_conf_pts(
                        v_conf_pt, t_conf_pt[ind_nn])
                    m_cost[ind_nn, ind_conf_pt] = m_cost[ind_conf_pt, ind_nn]

            if np.all(m_cost[ind_conf_pt, :] == np.inf):
                print(
                    f"Node {ind_conf_pt} is not connected with any other node."
                )

        return m_cost

    def _get_shortest_path_uav2_thru_pts_with_losNrcc_with_pts_with_losN2rcc_with_bs(
            self, bs_loc, l_ue_locs, uav2_start_loc=None):
        """
        args:
            + l_ue_locs: a (3,) numpy vector or a list of (3,) numpy vectors.
            + uav2_start_loc: a (3,) numpy vector or None. If None, the starting location of UAV2 is the nearest grid point to bs_loc.
        Returns:
            + l_shortest_path: a list of 1 x 3 arrays that contains the shortest
            trajectory of a UAV
                . starting at the bs_loc/uav2_start_loc, 
                . going through points that are in LOS with points that are in LOS with `bs_loc`, 
                . arriving at a location where UAV-2 can serve l_ue_locs. 
        """

        def get_locs_in_los_n_2rcc_with(bs_loc):
            """"
                Return a list of points that satisfy the following conditions:
                    + in LoS with BS,
                    + capacity from BS >= 2 * min_uav_rate
            """

            fly_grid = self._environment.fly_grid.clone()
            m_grid_pts = fly_grid.list_pts()

            l_locs = [
                m_grid_pts[ind]
                for ind in self._get_grid_pt_inds_in_los_n_minrate_with(bs_loc)
            ]

            return l_locs

        # Feasible positions for UAV1
        l_locs_uav1 = get_locs_in_los_n_2rcc_with(bs_loc)

        def is_uav2_conf_pt(pt, l_locs_uav1):
            """
                Returns true if pt can be a location for uav 2. This occurs when
                there exists A in l_feasible_pts_uav_1 s.t.
                    + pt and A are in LoS,
                    + capacity(A,pt) >= min_uav_rate
            """

            for loc_uav_1 in l_locs_uav1:
                if self.are_in_los(pt, loc_uav_1) and self.compute_capacity(
                        pt, loc_uav_1) >= self.min_uav_rate:
                    return True
            return False

        # can be unified with the one in greater_rate
        def min_ue_rate_grid_pt_inds(bs_loc, ue_loc, l_feasible_pts_uav_1,
                                     fly_grid_uav2):
            """It returns a list with the indices of the grid points where UAV2
            can be placed and such that there exists a point in
            `l_pts_in_los_with_bs` where UAV1 can be placed leading to a conf pt
            for which the user rate exceeds self.min_ue_rate. It is assumed that
            LOS is required at all links."""

            def can_get_min_ue_rate(v_pt_uav_2):
                for v_pt_uav_1 in l_feasible_pts_uav_1:
                    conf_pt = np.concatenate(
                        (v_pt_uav_1[None, :], v_pt_uav_2[None, :]), axis=0)

                    # We need to check if there is los connectivity between
                    # UAV1 and UAV2. Otherwise, a conf_pt may guarantee the
                    # min ue rate but there is no los connectivity between
                    # UAV1 and UAV2. This would cause the set of destinations of
                    # UAV1 to be empty because the trajectory of UAV1 requires
                    # los connectivity between the UAVs
                    if not self.are_in_los(v_pt_uav_1, v_pt_uav_2):
                        continue

                    if PathPlanner.conf_pt_to_is_ue_rate_gt_min(
                            self._channel,
                            conf_pt,
                            min_uav_rate=self.min_uav_rate,
                            min_ue_rate=self.min_ue_rate,
                            bs_loc=bs_loc,
                            ue_loc=ue_loc,
                            v_los=[True, True, True]):
                        return True
                return False

            m_grid_pts_uav_2 = fly_grid_uav2.list_pts()

            l_inds_los_with_ue = self._get_grid_pt_inds_in_los_with(
                ue_loc, fly_grid=fly_grid_uav2)

            return [
                ind_pt for ind_pt in l_inds_los_with_ue
                if can_get_min_ue_rate(m_grid_pts_uav_2[ind_pt])
            ]

        # Feasible points for uav2
        fly_grid_uav2 = self._environment.fly_grid.clone()
        # disable points that are not in (LoS and rcc) with uav1
        fly_grid_uav2.disable_by_indicator(
            lambda pt: not is_uav2_conf_pt(pt, l_locs_uav1))

        m_grid_pts_uav_2 = fly_grid_uav2.list_pts()
        t_conf_pt = m_grid_pts_uav_2[:, None, :]

        if uav2_start_loc is None:
            ind_start = fly_grid_uav2.nearest_ind(bs_loc)
        else:
            ind_start = fly_grid_uav2.nearest_ind(uav2_start_loc)

        if type(l_ue_locs) == np.ndarray:
            l_ue_locs = [l_ue_locs]

        if self._destination == "los":
            if len(l_ue_locs) > 1:
                raise NotImplementedError
            l_ind_end = self._get_grid_pt_inds_in_los_with(
                l_ue_locs[-1], fly_grid=fly_grid_uav2)

        elif self._destination == "nearest":
            if len(l_ue_locs) > 1:
                raise NotImplementedError
            l_ind_end = fly_grid_uav2.nearest_ind(l_ue_locs[-1])

        elif self._destination == "min_ue_rate":
            ls_ind_end = []
            for ue_loc in l_ue_locs:
                ls_ind_end.append(
                    set(
                        min_ue_rate_grid_pt_inds(bs_loc, ue_loc, l_locs_uav1,
                                                 fly_grid_uav2)))
            # compute the intersection of the sets in ls_ind_end
            for ind in range(len(ls_ind_end)):
                ls_ind_end_temp = ls_ind_end[ind:]
                l_ind_end = list(set.intersection(*ls_ind_end_temp))
                if len(l_ind_end) > 0:
                    l_ue_locs[:] = l_ue_locs[ind:]
                    break
        else:
            raise ValueError
        if len(l_ind_end) == 0:
            print("\nNo feasible trajectory of UAV2 found")
            return None

        m_cost = self._get_cost_all_cf_pts_2serve_static_ue(
            t_conf_pt, grid=fly_grid_uav2, neighbor_mode="grid")
        l_shortest_path_inds = self.get_shortest_path(m_cost,
                                                      ind_node_start=ind_start,
                                                      ind_nodes_end=l_ind_end)
        l_shortest_path = [t_conf_pt[ind] for ind in l_shortest_path_inds]

        return l_shortest_path

    def _get_shortest_path_uav_2_thru_pts_greater_rate_with_pts_greater_rate_with_bs(
            self, bs_loc, l_ue_locs, uav2_start_loc=None):
        """
        Returns:
            l_shortest_path: list of 1 x 3 arrays that contains the shortest
            trajectory of a UAV through points that have at least (r_cc + min_ue_rate) with points that have at least (2r_cc + min_ue_rate) with `bs_loc`. 
        """
        fly_grid = self._environment.fly_grid.clone()

        def min_ue_rate_grid_pt_inds(m_grid_pts_uav_2, bs_loc, ue_loc):
            """It returns a list with the indices of the grid points where UAV2
            can be placed and such that there exists a point in
            `l_pts_within_rate_with_bs` where UAV1 can be placed leading to a destination conf pt for which the user rate exceeds self.min_ue_rate. It is assumed that:
                + rate BS    -> UAV-1 >= 2*self.min_uav_rate.
                + rate UAV-1 -> UAV-2 >= self.min_uav_rate.
            """

            def can_get_min_ue_rate(v_pt_uav_2):
                for v_pt_uav_1 in l_pts_at_least_2rcc_with_bs:
                    conf_pt = np.concatenate(
                        (v_pt_uav_1[None, :], v_pt_uav_2[None, :]), axis=0)

                    # no need to check for LoS -> set v_los = [True, True, True]
                    if PathPlanner.conf_pt_to_is_ue_rate_gt_min(
                            self._channel,
                            conf_pt,
                            min_uav_rate=self.min_uav_rate,
                            min_ue_rate=self.min_ue_rate,
                            bs_loc=bs_loc,
                            ue_loc=ue_loc,
                            v_los=[False, False, False]):

                        return True
                return False

            l_inds_within_rate_with_ue = self._get_grid_pt_inds_in_minrate_with(
                ue_loc, self.min_ue_rate, fly_grid=fly_grid_uav2)

            return [
                ind_pt for ind_pt in l_inds_within_rate_with_ue
                if can_get_min_ue_rate(m_grid_pts_uav_2[ind_pt])
            ]

        def get_pts_at_least_rate_with_ref_pts(ref_pts,
                                               min_rate,
                                               fly_grid=None):

            assert fly_grid is not None

            m_grid_pts = fly_grid.list_pts()

            l_pts_at_least_rate_with_ref_pts = [
                m_grid_pts[ind]
                for ind in self._get_grid_pt_inds_in_minrate_with(
                    ref_pts, min_rate, fly_grid=fly_grid)
            ]
            return l_pts_at_least_rate_with_ref_pts

        # Feasible positions for UAV1
        l_pts_at_least_2rcc_with_bs = get_pts_at_least_rate_with_ref_pts(
            bs_loc, 2 * self.min_uav_rate, fly_grid=fly_grid)

        # Feasible points for UAV2
        fly_grid_uav2 = self._environment.fly_grid.clone()
        # disable points that dont have at least min_uav_rate with any point in
        # l_pts_at_least_2rcc_with_bs
        fly_grid_uav2.disable_by_indicator(
            lambda pt: not self.is_at_least_rate(
                pt, l_pts_at_least_2rcc_with_bs, self.min_uav_rate))

        m_grid_pts_uav_2 = fly_grid_uav2.list_pts()
        t_conf_pt = m_grid_pts_uav_2[:, None, :]

        if uav2_start_loc is None:
            ind_start = fly_grid_uav2.nearest_ind(bs_loc)
        else:
            ind_start = fly_grid_uav2.nearest_ind(uav2_start_loc)

        if type(l_ue_locs) == np.ndarray:
            l_ue_locs = [l_ue_locs]

        if self._destination == "los":
            if len(l_ue_locs) > 1:
                raise NotImplementedError
            l_ind_end = self.grid_pt_inds_in_los_with(l_ue_locs[-1],
                                                      grid=fly_grid_uav2)
        elif self._destination == "nearest":
            if len(l_ue_locs) > 1:
                raise NotImplementedError
            l_ind_end = fly_grid_uav2.nearest_ind(l_ue_locs[-1])
        elif self._destination == "min_ue_rate":
            ls_ind_end = []
            for ue_loc in l_ue_locs:
                ls_ind_end.append(
                    set(
                        min_ue_rate_grid_pt_inds(m_grid_pts_uav_2, bs_loc,
                                                 ue_loc)))
            # compute the intersection of the sets in ls_ind_end
            for ind in range(len(ls_ind_end)):
                ls_ind_end_temp = ls_ind_end[ind:]
                l_ind_end = list(set.intersection(*ls_ind_end_temp))
                if len(l_ind_end) > 0:
                    l_ue_locs[:] = l_ue_locs[ind:]
                    break
        else:
            raise ValueError

        if len(l_ind_end) == 0:
            print("No feasible trajectory of UAV2 found")
            return None

        m_cost = self._get_cost_all_cf_pts_2serve_static_ue(
            t_conf_pt, grid=fly_grid_uav2, neighbor_mode="grid")

        l_shortest_path_inds = self.get_shortest_path(m_cost,
                                                      ind_node_start=ind_start,
                                                      ind_nodes_end=l_ind_end)

        l_shortest_path = [t_conf_pt[ind] for ind in l_shortest_path_inds]

        return l_shortest_path

    def _get_cost_btw_conf_pts(self, conf_pt1, conf_pt2, loc_ue=None):
        """
        Args:
            - conf_pt1, conf_pt2: 2 x 3 arrays

        If loc_ue is
            - None: returns the maximum of the distances traveling from conf_pt1
              to conf_pt2 of the UAVs.

            - not None: returns 
            
                . -rate_ue provided by conf_pt2 if the grid points of conf_pt1
                are respectively adjacent to those of conf_pt2.

                . np.Inf otherwise.
        """

        def max_dist(pt1, pt2):
            return np.max(np.sqrt(np.sum((pt1 - pt2)**2, axis=1)))

        dist_grid_diag = np.ceil(
            np.sqrt(np.sum(self._environment.fly_grid.spacing**2)))

        if loc_ue is None:
            return max_dist(conf_pt1, conf_pt2)
        elif loc_ue is not None:
            if max_dist(conf_pt1, conf_pt2) <= dist_grid_diag:
                return -self.conf_pt_to_ue_rate(self._channel, conf_pt2,
                                                self.min_uav_rate, self.bs_loc,
                                                loc_ue)
            else:
                return np.Inf

        else:
            raise NotImplementedError

    def _concat_takeoff_and_landing(self, l_path_lifted, initial_pos_uav_2,
                                    final_pos_uav_2):
        spacing_z = self._environment.fly_grid.spacing[2]
        while np.any(initial_pos_uav_2 != l_path_lifted[0]):
            new_point = np.copy(l_path_lifted[0])
            new_point[0, 2] -= spacing_z
            l_path_lifted = [new_point] + l_path_lifted
        while np.any(final_pos_uav_2 != l_path_lifted[-1]):
            new_point = np.copy(l_path_lifted[-1])
            new_point[0, 2] -= spacing_z
            l_path_lifted = l_path_lifted + [new_point]
        return l_path_lifted

    def _get_path_both_uavs_given_path_uav2_2serve_static_ue(
            self, bs_loc, l_known_ue_locs, l_path_uav2, uav1_start_loc=None):
        """
            Plan the path of UAV-1 given the path of UAV-2 by lifting the path of UAV-2
        """

        def get_path_uav1_given_path_uav2_with_waiting(l_initial_path_uav_2,
                                                       bs_loc,
                                                       ue_loc,
                                                       uav1_start_loc=None):
            """
            Args:
                + ue_loc can be a (3,) numpy vector or a list of (3,) numpy vectors.
                + uav1_start_loc can be a (3,) numpy vector or None. If None, it is set to bs_loc.
            
            given the path of UAV2, it finds a suitable path for UAV1 so that
            UAV1 is all the time in LOS with both UAV2 and the BS. 

            To this end, this function may stop UAV2 at some of its waypoints to
            give time to UAV1 to move to the required position. 

            If no feasible UAV1 path exists, it returns None.        
            
            """

            def get_ind_nodes_end(v_eind_first_enode_each_block):
                """ It returns a vector with the einds that correspond to
                feasible final UAV1 grid points. It returns None if no feasible
                point exists.                 
                """

                if self.mode_connect == 'min_rate_only':
                    v_los = [False, False, False]
                else:
                    v_los = [True, True, True]

                def meets_ue_rate_constraint(eind):
                    """It is assumed that `pt_uav_2` is in LOS with the UE bc
                    the way the path of UAV1 was designed. """
                    ind_block, fly_grid_pt_ind = self._eind_to_block_and_fly_grid_pt_ind(
                        eind, ll_uav1_fea_inds_at_each_uav2_wpts)
                    pt_uav_1 = m_grid_pts[fly_grid_pt_ind]
                    pt_uav_2 = l_initial_path_uav_2[ind_block][0]
                    conf_pt = np.concatenate(
                        (pt_uav_1[None, :], pt_uav_2[None, :]), axis=0)

                    # TODO: modify to consider the case of l_ue_locs, more
                    # realizations are required to test how likely the current
                    # implementation is feasible.

                    l_b_meet = []
                    if type(ue_loc) == np.ndarray:
                        l_ue_locs = [ue_loc]
                    else:
                        l_ue_locs = ue_loc
                    for temp_ue_loc in l_ue_locs:
                        l_b_meet.append(
                            PathPlanner.conf_pt_to_is_ue_rate_gt_min(
                                self._channel,
                                conf_pt,
                                min_uav_rate=self.min_uav_rate,
                                min_ue_rate=self.min_ue_rate,
                                bs_loc=bs_loc,
                                ue_loc=temp_ue_loc,
                                v_los=v_los))

                        # TODO: what if only some of l_ue_locs are True?

                    return np.all(l_b_meet)

                l_einds_feasible_for_uav_2_and_bs = np.arange(
                    v_eind_first_enode_each_block[-2],
                    v_eind_first_enode_each_block[-1])
                if self._destination in {"los", "nearest"}:
                    return l_einds_feasible_for_uav_2_and_bs
                elif self._destination == "min_ue_rate":
                    l_einds = np.array([
                        eind for eind in l_einds_feasible_for_uav_2_and_bs
                        if meets_ue_rate_constraint(eind)
                    ])
                    if len(l_einds) == 0:
                        # Since we entered this function, it means that a
                        # feasible trajectory for UAV2 was found. In particular,
                        # a feasible destination was found. Thus, this list
                        # should not be empty.

                        raise ValueError
                    return l_einds

            def get_m_cost_extended_graph(fly_grid,
                                          ll_uav_fea_inds_vs_time,
                                          b_diagonal=True):
                """
                    Args:
                        + If b_diagonal is True, the costs in the diagonal will be computed, which means that the UAV can move from one node to another in the same block, else the diagonal will be np.Inf.
                """

                # Total number of nodes
                num_blocks = len(ll_uav_fea_inds_vs_time)
                m_grid_pts = fly_grid.list_pts()

                num_ex_nodes = 0
                s_inds = set(ll_uav_fea_inds_vs_time[0])
                for l_inds in ll_uav_fea_inds_vs_time:
                    num_ex_nodes += len(l_inds)
                    s_inds = s_inds.union(l_inds)

                ind_max = np.max(list(s_inds))
                m_stored_cost = np.zeros((ind_max + 1, ind_max + 1))
                m_stored_cost[:] = np.nan

                m_costs = np.inf * np.ones((num_ex_nodes, num_ex_nodes))

                def get_block(l_inds_1, l_inds_2):
                    """
                    It returns a matrix whose [i,j]-th entry is 1 if l_inds_1[i] and
                    l_inds_2[j] are adjacent and np.Inf otherwise.
                    """

                    def get_cost_for_inds(ind_1, ind_2):
                        if not np.isnan(m_stored_cost[ind_1, ind_2]):
                            return m_stored_cost[ind_1, ind_2]
                        else:
                            if ind_1 == ind_2:
                                m_stored_cost[ind_1, ind_2] = 0.
                                return m_stored_cost[ind_1, ind_2]

                            is_adjacent = fly_grid.are_adjacent(
                                m_grid_pts[ind_1], m_grid_pts[ind_2])

                            if is_adjacent:
                                m_stored_cost[ind_1, ind_2] = np.linalg.norm(
                                    m_grid_pts[ind_1] - m_grid_pts[ind_2])
                            else:
                                m_stored_cost[ind_1, ind_2] = np.Inf

                            return m_stored_cost[ind_1, ind_2]

                    m_block = np.array([[
                        get_cost_for_inds(ind_1, ind_2) for ind_2 in l_inds_2
                    ] for ind_1 in l_inds_1])

                    return m_block

                ind_block_start_row = 0

                for ind_block in range(
                        num_blocks):  #Iterates along block rows of m_costs

                    ind_block_end_row = ind_block_start_row + len(
                        ll_uav_fea_inds_vs_time[ind_block])

                    # Block diagonal
                    if b_diagonal:
                        m_costs[
                            ind_block_start_row:ind_block_end_row,
                            ind_block_start_row:ind_block_end_row] = get_block(
                                ll_uav_fea_inds_vs_time[ind_block],
                                ll_uav_fea_inds_vs_time[ind_block])

                    if ind_block < num_blocks - 1:
                        # Block diagonal right above the main block diagonal

                        ind_block_end_next_row = ind_block_end_row + len(
                            ll_uav_fea_inds_vs_time[ind_block + 1])

                        m_costs[ind_block_start_row:ind_block_end_row,
                                ind_block_end_row:
                                ind_block_end_next_row] = get_block(
                                    ll_uav_fea_inds_vs_time[ind_block],
                                    ll_uav_fea_inds_vs_time[ind_block + 1])

                    ind_block_start_row = ind_block_end_row

                return m_costs

            fly_grid = self._environment.fly_grid.clone()
            m_grid_pts = fly_grid.list_pts()

            # We construct an extended graph. Each node of the extended graph
            # (enode) corresponds to a fly grid pt index and a block index.
            if self.mode_connect == 'min_rate_only':
                s_grid_pts_inds_at_least_rate_with_bs = set(
                    self._get_grid_pt_inds_in_minrate_with(
                        bs_loc, 2 * self.min_uav_rate))
                ll_uav1_fea_inds_at_each_uav2_wpts = [
                    list(
                        s_grid_pts_inds_at_least_rate_with_bs.intersection(
                            set(
                                self._get_grid_pt_inds_in_minrate_with(
                                    pt, self.min_uav_rate))))
                    for pt in l_initial_path_uav_2
                ]

            else:
                # 0.  Find the list of lists of intersections between the set of fly grid points in LoS with the BS and the sets of fly grid points in LoS with waypoints of the trajectory of uav 2.
                s_grid_pt_inds_fea_uav_1 = set(
                    self._get_grid_pt_inds_in_los_n_minrate_with(bs_loc))

                ll_uav1_fea_inds_at_each_uav2_wpts = [
                    list(
                        s_grid_pt_inds_fea_uav_1.intersection(
                            self._get_grid_pt_inds_in_los_n_minrate_with(
                                pt[0], min_rate=self.min_uav_rate)))
                    for pt in l_initial_path_uav_2
                ]

            # 0. Find the eind of the first enode of each block
            v_eind_first_enode_each_block = PathPlanner.get_eind_first_enode_each_block(
                ll_uav1_fea_inds_at_each_uav2_wpts)

            # 1. Construct m_cost
            m_cost = get_m_cost_extended_graph(
                fly_grid, ll_uav1_fea_inds_at_each_uav2_wpts, b_diagonal=True)

            # 2. Invoke shortest_path()
            if uav1_start_loc is None:
                uav1_start_loc = bs_loc
            ind_enode_start = self._block_and_fly_grid_pt_inds_to_einds(
                0, fly_grid.nearest_ind(uav1_start_loc),
                ll_uav1_fea_inds_at_each_uav2_wpts)[0]
            ind_nodes_end = get_ind_nodes_end(v_eind_first_enode_each_block)
            if ind_nodes_end is None:
                return None

            l_path_einds = self.get_shortest_path(
                m_cost,
                ind_node_start=ind_enode_start,
                ind_nodes_end=ind_nodes_end)
            if l_path_einds is None:
                return None

            # 3. Find where UAV2 needs to wait and repeat the positions in
            #    `l_initial_path_uav_2` accordingly
            l_path_ind_block_ind_grid_pt = [
                self._eind_to_block_and_fly_grid_pt_ind(
                    eind, ll_uav1_fea_inds_at_each_uav2_wpts)
                for eind in l_path_einds
            ]
            l_path_uav_2 = [
                l_initial_path_uav_2[ind_block_ind_grid_pt[0]]
                for ind_block_ind_grid_pt in l_path_ind_block_ind_grid_pt
            ]

            # 4. Combine and return the trajectories of both UAVs.
            l_path_uav_1 = [
                m_grid_pts[ind_block_ind_grid_pt[1], None, :]
                for ind_block_ind_grid_pt in l_path_ind_block_ind_grid_pt
            ]
            lm_path = [
                np.concatenate((pos_uav_1, pos_uav_2), axis=0)
                for pos_uav_1, pos_uav_2 in zip(l_path_uav_1, l_path_uav_2)
            ]

            return lm_path

        initial_pos_uav2, final_pos_uav2 = l_path_uav2[0], l_path_uav2[-1]
        while True:
            lm_path_uavs = get_path_uav1_given_path_uav2_with_waiting(
                self._concat_takeoff_and_landing(l_path_uav2, initial_pos_uav2,
                                                 final_pos_uav2),
                bs_loc,
                l_known_ue_locs,
                uav1_start_loc=uav1_start_loc)

            if lm_path_uavs is not None:
                return lm_path_uavs

            # lift the path of UAV-2
            l_path_uav2 = self._lift_path(l_path_uav2)

            if l_path_uav2 is None:
                raise ValueError(
                    "No path can be found even by lifting the path of UAV2. Most likely some buildings are higher than the highest grid point. "
                )

    def _sample_around_tentative_path(self,
                                      lm_feas_path,
                                      l_feas_uav1,
                                      l_feas_uav2,
                                      lm_ue_path=None):
        """
        Args:

        - lm_ue_path: either None or list of 1 x 3 arrays with the same length
          as lm_feas_path. 


        If lm_ue_path is:
        
            + None, returns a list of 2 x 3 arrays of configuration points. The
              length of this list is determined by self._num_nodes. (Used for a
              static UE)

            + not None, returns a list of lists of 2 x 3 arrays. The n-th list 
            contains the configuration points around the n-th point of
            lm_feas_path. So the number of lists equals the number of conf.
            points in lm_feas_path. (Used for a moving UE)

        Each drawn conf. points meets a certain condition, which is determined
        by self.mode_connect and self.b_conf_pts_meet_min_ue_rate.   

        """

        def sample_around_conf_pt(ref_conf_pt,
                                  l_feas_uav1,
                                  l_feas_uav2,
                                  num_pts,
                                  ue_loc=None):
            """
            Draws num_pts configuration points around ref_conf_pt. For each
            configuration point, the locations of UAV1 and UAV2 are drawn from
            l_feas_uav1 and l_feas_uav2, respectively with probabilities
            inversely proportional to the disance from the candidate points in
            these lists to ref_conf_pt.

            The drawn conf. points meet a certain condition, which is determined
            by self.mode_connect and self.b_conf_pts_meet_min_ue_rate.         

            If a point is in l_feas_uavN, it is expected that it is not inside a
            building. 
            """

            ref_loc_uav1, ref_loc_uav2 = ref_conf_pt

            def sample_around_uav_loc(loc_uav, l_feas_pts):
                """
                Create a generator that randomly chooses a location in
                l_feas_pts with a probability that is inversely proportional to
                its distance to loc_uav.
                """

                def get_dist(pt_1, pt_2):
                    dist = np.linalg.norm(pt_1 - pt_2)
                    if dist == 0:
                        return 1e9
                    else:
                        return dist

                l_dist_to_uav = [
                    get_dist(loc_uav, feas_pts) for feas_pts in l_feas_pts
                ]
                v_probs = 1 / np.array(l_dist_to_uav)
                v_probs = v_probs / np.sum(v_probs)

                while True:
                    yield l_feas_pts[np.random.choice(np.arange(
                        len(l_feas_pts)),
                                                      p=v_probs)]

            g_pts_uav1 = sample_around_uav_loc(ref_loc_uav1, l_feas_uav1)
            g_pts_uav2 = sample_around_uav_loc(ref_loc_uav2, l_feas_uav2)

            lm_conf_pts = []
            for pt_uav1, pt_uav2 in zip(g_pts_uav1, g_pts_uav2):

                m_conf_pt = np.concatenate(
                    (pt_uav1[np.newaxis, ...], pt_uav2[np.newaxis, ...]),
                    axis=0)

                # exit if we draw enough points
                if len(lm_conf_pts) >= num_pts:
                    return lm_conf_pts

                if self.b_conf_pts_meet_min_ue_rate:
                    assert ue_loc is not None
                    if self._is_in_Qfree(m_conf_pt, ue_loc):
                        lm_conf_pts.append(m_conf_pt)
                else:
                    if self._is_in_Qfree(m_conf_pt):
                        lm_conf_pts.append(m_conf_pt)

        lm_conf_pts = []
        num_pts_per_feas_pt = np.floor(
            (self._num_nodes - len(lm_feas_path)) / len(lm_feas_path))

        # For each configuration point in lm_feas_path, draw num_pts_per_feas_pt
        # configuration points
        for ind, conf_pt in enumerate(lm_feas_path):

            ue_loc = lm_ue_path[ind][0] if lm_ue_path is not None else None

            # add the current conf. point of lm_feas_path to lm_conf_pts to make
            # sure that there always exists a path in lm_conf_pts
            lm_current_pts = [lm_feas_path[ind]] + sample_around_conf_pt(
                conf_pt,
                l_feas_uav1,
                l_feas_uav2,
                num_pts=num_pts_per_feas_pt,
                ue_loc=ue_loc)

            if lm_ue_path is not None:
                lm_conf_pts.append(lm_current_pts)
            else:
                lm_conf_pts.extend(lm_current_pts)

        return lm_conf_pts

    @staticmethod
    def _eind_to_block_and_fly_grid_pt_ind(eind, ll_pts):
        """
        Args:
            + ll_pts can be list of lists of (indices or grid points or configuration points).

        Returns:
            + ind_block: index of the block that contains the node with index eind.

            + pts: the index, grid point, or configuration point (2x3) that has index eind in the block ind_block.
        """

        v_eind_first_enode_each_block = PathPlanner.get_eind_first_enode_each_block(
            ll_pts)
        ind_block = np.where(eind -
                             v_eind_first_enode_each_block < 0)[0][0] - 1
        ind_within_block = eind - v_eind_first_enode_each_block[ind_block]
        pts = ll_pts[ind_block][ind_within_block]

        return ind_block, pts

    @staticmethod
    def _block_and_fly_grid_pt_inds_to_einds(ind_block, l_fly_grid_pt_inds,
                                             ll_fea_inds):
        """
        Args:
            + l_fly_grid_pt_inds: can be scalar or list. If it is                
                . a scalar, returns the index of the corresponding enode,
                . a list, returns a list of indices of the corresponding enodes.
        """

        v_eind_first_enode_each_block = PathPlanner.get_eind_first_enode_each_block(
            ll_fea_inds)

        if type(l_fly_grid_pt_inds) == np.int64:
            l_fly_grid_pt_inds = [l_fly_grid_pt_inds]

        l_enode_inds = list(v_eind_first_enode_each_block[ind_block] + [
            np.where(
                np.array(ll_fea_inds[ind_block]) == fly_grid_pt_inds)[0][0]
            for fly_grid_pt_inds in l_fly_grid_pt_inds
        ])

        return l_enode_inds

    def _lift_path(self, l_path):
        """
        The points in l_path are lifted to the points right above them if
        they are inside the grid, else they remain the same. The first and
        last points of l_path are respectively prepended and appended at the
        end if they can be lifted. 
        
        Exception: if l_path == l_path_lifted, this function returns None.

        Args: - `l_path`: list of 1 x 3 arrays with grid points. 
        
        Returns:

        - `l_path_lifted`
        
        """
        fly_grid = self._environment.fly_grid
        spacing_z = fly_grid.spacing[2]
        has_changed = False

        def lift_point(pt_in):
            nonlocal has_changed
            pt_out = np.copy(pt_in)
            pt_out[2] += spacing_z
            if fly_grid.is_within_limits(pt_out):
                has_changed = True
                return pt_out
            else:
                return pt_in

        l_path_lifted = [lift_point(pt[0])[None, :] for pt in l_path]

        if not has_changed:
            return None
        return l_path_lifted

    def _is_in_Qfree(self, m_conf_pt, loc_ue=None):
        """
        If self.mode_connect is

            (1) 'min_rate_only': 

                - If loc_ue is not None, returns True iff the rate of the UE is
                  at least self.min_ue_rate.

                - Else, returns True iff the UAV at m_conf_pt[-1] receives at
                  least self.min_uav_rate.

            (2) 'los_n_rate':

                - If loc_ue is not None, returns True iff (1) and there is los
                  from the bs -> uav1 -> uav2 -> ... -> uavK -> the ue.

                - Else, returns True iff (1) and there is los from the bs ->
                  uav1 -> uav2 -> ... -> uavK.
            
            else: raise NotImplementedError
        """

        num_uavs = m_conf_pt.shape[0]

        def check_los():
            for ind_uav in range(num_uavs):
                if ind_uav == 0:
                    node_prev = self.bs_loc
                else:
                    node_prev = m_conf_pt[ind_uav - 1]
                if not self.are_in_los(node_prev, m_conf_pt[ind_uav]):
                    return False

            if loc_ue is not None:
                if not self.are_in_los(m_conf_pt[-1], loc_ue):
                    return False
            return True

        def check_min_rate():
            for ind_uav in range(num_uavs):
                if ind_uav == 0:
                    rate_current_uav = self.compute_capacity(
                        self.bs_loc, m_conf_pt[0])
                else:
                    rate_from_prev_uav = rate_current_uav - self.min_uav_rate
                    capacity_between_prev_uav = self.compute_capacity(
                        m_conf_pt[ind_uav - 1], m_conf_pt[ind_uav])
                    rate_current_uav = np.maximum(
                        0,
                        np.minimum(rate_from_prev_uav,
                                   capacity_between_prev_uav))
                # return False if the rate received from the previous UAV is not
                # enough to provide min_uav_rate to the current and the rest
                # UAVs
                if rate_current_uav < (num_uavs - ind_uav) * self.min_uav_rate:
                    return False

            if loc_ue is not None:
                rate_ue = np.maximum(
                    0,
                    np.minimum(rate_current_uav - self.min_uav_rate,
                               self.compute_capacity(m_conf_pt[-1], loc_ue)))
                if rate_ue < self.min_ue_rate:
                    return False
            return True

        if self.mode_connect == 'min_rate_only':
            return check_min_rate()
        elif self.mode_connect == 'los_n_rate':
            return check_los() and check_min_rate()
        else:
            raise NotImplementedError

    def _are_connected(self, m_conf_pt_1, m_conf_pt_2):
        """
        Returns True if the following conditions hold:

        C1: There is LOS between m_conf_pt_1[ind_uav] and m_conf_pt_2[ind_uav]
        for all ind_uav (this guarantees that the UAVs can move from m_conf_pt_1
        to m_conf_pt_2 in a straight line without colliding with a building)

        C2: a set of points on the line between m_conf_pt_1 and m_conf_pt_2
        (uniformly spaced by a distance such that all UAVs move no more than
        self._step_connectivity between each two consecutive points) are in Qfree. 

        It is assumed that `m_conf_pt_1` and `m_conf_pt_2` are in Qfree.
        """

        m_direction = m_conf_pt_2 - m_conf_pt_1
        max_uav_dist = np.max(np.linalg.norm(m_conf_pt_1 - m_conf_pt_2,
                                             axis=1))
        if (max_uav_dist == 0):
            return True

        # the UAVs cannot go through buildings
        # C1:
        num_uavs = m_conf_pt_1.shape[0]
        if not np.all([
                self.are_in_los(m_conf_pt_1[ind_uav], m_conf_pt_2[ind_uav])
                for ind_uav in range(num_uavs)
        ]):
            return False

        # C2:
        m_direction = m_direction / max_uav_dist  # the largest row has norm 1
        num_steps = int(np.floor(max_uav_dist / self._step_connectivity))
        for ind_step in range(1, num_steps):
            # we already check the constraints on min_uav_rate among BS, UAV1, and UAV2 in the following step
            if not self._is_in_Qfree(m_conf_pt_1 + ind_step *
                                     self._step_connectivity * m_direction):
                return False
        return True


class SegmentationPathPlanner(RandomRoadmapPathPlanner):

    _name_on_figs = "Benchmark 4"

    integral_mode = 'c'  # Can be 'c' or 'python'

    def __init__(self, num_known_ue_locs=10, num_locs_to_replan=5, **kwargs):

        super().__init__(**kwargs)
        self.num_known_ue_locs = num_known_ue_locs
        self.num_locs_to_replan = num_locs_to_replan

    def plan_path_to_serve_moving_ue(self, bs_loc, lm_ue_path, debug=False):
        """
        Overview: For every (num_locs_to_replan) time steps, the UAVs are given
        the next (num_known_ue_locs) locations of the user, num_locs_to_replan
        <= num_known_ue_locs. The following steps will be iteratively
        implemented at time steps (n_t * num_locs_to_replan), n_t is an integer.
            1. At time step (n_t * num_locs_to_replan), the planner uses
            PathPlanner.plan_path_to_serve_static_ue (PRFI) (*) to plan a path
            for the UAVs from given start grid points to the nearest grid points
            where they can serve 
                . all of the next (num_known_ue_locs) locations of the user.

                . If such grid points do not exist, the planner plans a path to
                the nearest grid points where the UAVs can serve the last
                (num_known_ue_locs - 1) known locations of the user, i.e.,
                lm_ue_path[n_t * num_locs_to_replan + n], n = 1,...,
                num_known_ue_locs - 1.

                . and so on.

                . In the most extreme case when the UAVs cannot find grid points
                to simultaneously guarantee min_ue_rate to multiple
                locations of the user, the planner plans a path to the nearest
                grid point where the UAVs can serve the last known location of
                the user, i.e., lm_ue_path[n_t * num_locs_to_replan +
                num_known_ue_locs - 1].
                 
            Noted that the user is currently at lm_ue_path[n_t *
            num_locs_to_replan]. 

            (*) The planner here just uses the tentative path provided by the
            PRFI.

            2. If the length of the path obtained in 1. is less than
            (num_locs_to_replan), the last configuration point of the path is
            repeated until the length of the path is (num_locs_to_replan). If
            the length of the path obtained in 1. is greater than
            (num_locs_to_replan), only the first (num_locs_to_replan)
            configuration points of the path are kept. 
            
            3. The last configuration point of the path obtained in 2. provides
               the start locations of the UAVs in the next iteration, i.e., at
               time step ((n_t + 1) * num_locs_to_replan).        
        
        Args:
            - lm_ue_path: a list of (1, 3) matrices containing locations of the
            user.

            - num_known_ue_locs: the number of locations of the user known in
              every
            time step.

            - num_locs_to_replan: the number of locations of the user to replan
              the
            path of the UAVs.        

        """
        print(f'    o {self.name}')

        assert self.num_locs_to_replan <= self.num_known_ue_locs
        l_replan_loc_inds = np.arange(0, len(lm_ue_path),
                                      self.num_locs_to_replan)
        lm_path_concat = [
            np.concatenate((bs_loc[None, :], bs_loc[None, :]), axis=0)
        ]

        for ind_replan, current_replan_ind in enumerate(l_replan_loc_inds):

            if debug:
                print(f"Planning {ind_replan}/{len(l_replan_loc_inds)}")

            # known locations of the user
            l_known_ue_locs = [
                m_loc[0]
                for m_loc in lm_ue_path[current_replan_ind:current_replan_ind +
                                        self.num_known_ue_locs]
            ]

            # find the starting locations of UAV-1 and UAV-2, which change over time
            uav1_start_loc = lm_path_concat[-1][0]
            uav2_start_loc = lm_path_concat[-1][1]
            lm_path_uavs = [lm_path_concat[-1]]

            # plan path for UAV-2
            l_path_uav2 = self._get_shortest_path_uav_2_thru_pts_greater_rate_with_pts_greater_rate_with_bs(
                bs_loc, l_known_ue_locs, uav2_start_loc=uav2_start_loc)

            if l_path_uav2 is not None:
                # plan path of both UAVs, given the path of UAV-2, included lifting and waiting in the path of UAV-2
                lm_path_uavs = self._get_path_both_uavs_given_path_uav2_2serve_static_ue(
                    bs_loc, l_known_ue_locs, l_path_uav2, uav1_start_loc)

            # concatenate the path of the uavs
            for ind in range(self.num_locs_to_replan):
                if ind < len(lm_path_uavs):
                    m_locs_uavs = lm_path_uavs[ind]
                else:
                    m_locs_uavs = lm_path_uavs[-1]

                if len(lm_path_concat) >= len(lm_ue_path):
                    break

                lm_path_concat.append(m_locs_uavs)

        return lm_path_concat
