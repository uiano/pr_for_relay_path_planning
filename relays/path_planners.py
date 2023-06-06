import random
import sys

import matplotlib.pyplot as plt
import numpy as np

from gsim_conf import use_mayavi
from common.fields import FunctionVectorField
import copy


class PathPlanner():

    # to be overridden by subclasses
    _name_on_figs = ""

    def __init__(self, environment, channel, pars_on_name=[]):
        """
        
        Args:

            - `bs_loc`: (N,3)-array-like where N is the number of ground BSs.

            - `pars_on_name`: list of tuples (str_format, par_name) used by __str__. 

        """

        self._environment = environment
        self._channel = channel
        self.pars_on_name = pars_on_name

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
            for ind_step in range(1, num_steps)
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

    def plan_path(self, bs_loc, ue_loc, samp_int, max_uav_speed,
                  uav_loc_start):
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
    def rate_from_path(channel, lm_path, min_uav_rate, bs_loc, ue_loc=None):
        """
            Returns:
                + m_uavs_rate: num_uavs x num_time_step matrix

                IF ue_loc is not None, it also returns

                + v_ue_rate:   vector of length num_time_step
        """

        num_time_step = len(lm_path)
        num_uavs = lm_path[0].shape[0]
        m_uavs_rate = np.zeros((num_uavs, num_time_step))
        v_user_rate = np.zeros((num_time_step, ))

        def compute_capacity(pt1, pt2):
            return channel.dbgain_to_capacity(channel.dbgain(pt1, pt2))

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

            if ue_loc is not None:
                # Compute users's rate
                if m_uavs_rate[-1, ind_step] >= min_uav_rate:
                    v_user_rate[ind_step] = np.minimum(
                        m_uavs_rate[-1, ind_step] - min_uav_rate,
                        compute_capacity(lm_path[ind_step][-1], ue_loc))
                else:
                    v_user_rate[ind_step] = 0

        if ue_loc is not None:
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
        return self.__class__.__name__

    @property
    def name_on_figs(self):
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
                                           ue_loc=ue_loc)
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

    def plot_path(self,
                  path,
                  interval=100,
                  scale_factor=10.,
                  azim=None,
                  elev=None,
                  **kwargs):
        """
        Args:

            -`path`: len_path x num_uavs x 3

            -`interval`: num of ms per frame

        Note: returned value should be stored in a variable. E.g.:

        anim = obj.plot_path(path)
        plt.plot()
        
        """

        import matplotlib.animation as animation

        if path is None:
            print("plot_path received no path to plot.")

        env = self._environment
        path = np.array(path)
        ax = env.plot(**kwargs)

        if use_mayavi:
            from mayavi import mlab

            t_path = np.array(path)
            num_time_slots, num_uavs, _ = t_path.shape

            # l_colors = [[.4, .8, .0], [.7, 0, 0], [.7, .3, .1], [0, .5, .9],
            #             [.8, .0, .1]]
            l_colors = [[.4, .8, .0], [1, 1, 0], [.7, .3, .1], [0, .5, .9],
                        [.8, .0, .1]]
            l_markers = [
                "2dcircle",
                "2dcross",
                "2ddiamond",
                "2ddash",
                "2dhooked_arrow",
                "2dsquare",
                "2dthick_arrow",
                "2dthick_cross",
                "2dtriangle",
                "2dvertex",
                "arrow",
                "axes",
                "cone",
                "cube",
                "cylinder",
                "point",
                "sphere",
                "2darrow",
            ]

            l_uav_pos = []
            ind_time_slot = 0
            for ind_uav in range(num_uavs):
                uav_pos = mlab.points3d([t_path[ind_time_slot, ind_uav, 0]],
                                        [t_path[ind_time_slot, ind_uav, 1]],
                                        [t_path[ind_time_slot, ind_uav, 2]],
                                        mode=l_markers[ind_uav],
                                        scale_factor=scale_factor,
                                        color=tuple(l_colors[ind_uav]))
                l_uav_pos.append(uav_pos)

            mlab.view(azimuth=azim, elevation=elev)

            @mlab.animate(delay=250)
            def anim():

                for ind_time_slot in range(
                        1, num_time_slots
                ):  # this may be the number of time slots
                    for ind_uav in reversed(range(num_uavs)):
                        v_x = t_path[ind_time_slot - 1:ind_time_slot + 1,
                                     ind_uav, 0]
                        v_y = t_path[ind_time_slot - 1:ind_time_slot + 1,
                                     ind_uav, 1]
                        v_z = t_path[ind_time_slot - 1:ind_time_slot + 1,
                                     ind_uav, 2]

                        mlab.plot3d(
                            v_x,
                            v_y,
                            v_z,
                            #representation='points',
                            color=tuple(l_colors[ind_uav]),
                            tube_radius=.5)

                        l_uav_pos[ind_uav].mlab_source.trait_set(
                            x=[t_path[ind_time_slot, ind_uav, 0]],
                            y=[t_path[ind_time_slot, ind_uav, 1]],
                            z=[t_path[ind_time_slot, ind_uav, 2]],
                            # mode=l_markers[ind_uav],
                            # scale_factor=scale_factor,
                            # color=tuple(l_colors[ind_uav])
                        )

                    # fig.scene.reset_zoom()
                    yield

            return anim()
        else:
            fig = ax.figure
            num_pts, num_uavs, _ = path.shape
            l_lines = []
            for ind_uav in range(num_uavs):
                line, = ax.plot([], [], [])
                l_lines.append(line, )

            def init():
                return ax.artists

            def run(ind_pt):
                for ind_uav, line in enumerate(l_lines):
                    line.set_data(path[:ind_pt, -1 - ind_uav, 0],
                                  path[:ind_pt, -1 - ind_uav, 1])
                    line.set_3d_properties(path[:ind_pt, -1 - ind_uav, 2])
                return l_lines

            anim = animation.FuncAnimation(fig,
                                           run,
                                           frames=np.arange(num_pts),
                                           interval=interval,
                                           init_func=init,
                                           save_count=100,
                                           blit=True)

            return anim

    def combine_paths(lm_path_1, lm_path_2):
        """
        Repeats the last entry of the shortest of these two lists to make both
        lists of the same length. Then, it concatenates the entries of both
        lists into a single list. 

        """
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

    def plan_path(self,
                  bs_loc,
                  ue_loc,
                  samp_int,
                  max_uav_speed,
                  uav_loc_start=None):

        if uav_loc_start is None:
            uav_loc_start = bs_loc

        # trajectory without min_uav_rate
        l_loc_vs_time, _, _ = self._path_w_takeoff(uav_loc_start,
                                                   (bs_loc + ue_loc) / 2,
                                                   samp_int, max_uav_speed)
        lm_path = [loc[None, :] for loc in l_loc_vs_time]

        lm_path = self.constrain_path_with_rmin(lm_path, self.min_uav_rate,
                                                bs_loc)

        return lm_path


class TwoRelaysAbovePathPlanner(PathPlanner):

    _name_on_figs = "Benchmark 2"

    def __init__(self, fly_height=40, min_uav_rate=None, **kwargs):

        super().__init__(**kwargs)
        self.fly_height = fly_height
        self.min_uav_rate = min_uav_rate

    def plan_path(self,
                  bs_loc,
                  ue_loc,
                  samp_int,
                  max_uav_speed,
                  uav_loc_start=None):

        if uav_loc_start == None:
            uav_loc_start = bs_loc

        def plan_path_without_rmin():
            l_loc_vs_time, num_takeoff_int, num_fly_int = self._path_w_takeoff(
                uav_loc_start, ue_loc, samp_int, max_uav_speed)

            one_third_bs2ue = int(np.ceil((num_fly_int / 3)))
            num_int_uav1 = num_takeoff_int + one_third_bs2ue
            num_int_uav2 = num_takeoff_int + 2 * one_third_bs2ue

            lm_path = []

            for ind_time in range(num_int_uav2):
                if ind_time <= num_int_uav1:
                    uav1_loc = l_loc_vs_time[ind_time]
                else:
                    uav1_loc = l_loc_vs_time[num_int_uav1]

                uav2_loc = l_loc_vs_time[ind_time]

                lm_path.append(np.array([uav1_loc, uav2_loc]))

            return lm_path

        # at this point, we have the path without the constraint on min_uav_rate
        lm_path = plan_path_without_rmin()

        lm_path = self.constrain_path_with_rmin(lm_path, self.min_uav_rate,
                                                bs_loc)

        return lm_path


class UniformlySpreadRelaysPathPlanner(PathPlanner):
    """ Benchmark 3"""

    _name_on_figs = "Benchmark 3"

    def __init__(self, num_uavs=4, min_uav_rate=None, fly_height=40, **kwargs):

        super().__init__(**kwargs)
        self.num_uavs = num_uavs
        self.min_uav_rate = min_uav_rate
        self.fly_height = fly_height

    def plan_path(self,
                  bs_loc,
                  ue_loc,
                  samp_int,
                  max_uav_speed,
                  uav_loc_start=None):

        if uav_loc_start == None:
            uav_loc_start = bs_loc

        def plan_path_without_rmin():

            lm_path = []

            v_is_update = [True] * self.num_uavs

            l_loc_vs_time, num_takeoff_int, num_fly_int = self._path_w_takeoff(
                uav_loc_start, ue_loc, samp_int, max_uav_speed)

            m_uav_loc = np.array(l_loc_vs_time)
            total_time_steps = num_takeoff_int + num_fly_int

            dis_in_time_steps_btw_uav = int(
                np.ceil(num_fly_int / (self.num_uavs - 1)))

            m_uav_des = m_uav_loc[num_takeoff_int::dis_in_time_steps_btw_uav -
                                  1]

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

        lm_path = plan_path_without_rmin()

        if self.min_uav_rate is not None:
            lm_path = self.constrain_path_with_rmin(lm_path, self.min_uav_rate,
                                                    bs_loc)

        return lm_path


class RandomRoadmapPathPlanner(PathPlanner):

    _name_on_figs = "PRFI"  # probabilistic path planner

    integral_mode = 'c'  # Can be 'c' or 'python'

    def __init__(self,
                 num_uavs=2,
                 num_nodes=1000,
                 max_num_neighbors=20,
                 step_connectivity=None,
                 min_uav_rate=None,
                 fly_height=None,
                 des_coor=None,
                 mode_draw_conf_pt="grid",
                 mode_connect="los",
                 destination="los",
                 min_ue_rate=None,
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

    def plan_path(self, bs_loc, ue_loc, samp_int, max_uav_speed):

        self.bs_loc = bs_loc

        lm_path_coarse = self.plan_path_coarse(bs_loc, ue_loc)

        lm_path = self.resample(lm_path_coarse, samp_int, max_uav_speed)

        return lm_path

    def plan_path_coarse(self, bs_loc, ue_loc):

        self.bs_loc = bs_loc

        m_uav_loc_start = np.tile(bs_loc, (self._num_uavs, 1))
        m_uav_loc_end = np.tile(ue_loc, (self._num_uavs, 1))
        m_uav_loc_end[:, 2] = self.fly_height

        # Draw configuration points
        t_conf_pt = self.get_conf_pts(bs_loc,
                                      ue_loc,
                                      mode=self.mode_draw_conf_pt)
        if t_conf_pt is None:

            print('RR: t_conf_pt is None, return None')
            return None

        # Add the initial locations to make sure that it is possible to go from
        # the bs to the first configuration pioint.
        t_conf_pt = np.concatenate((m_uav_loc_start[None, :], t_conf_pt),
                                   axis=0)

        # Compute the cost
        m_cost = self.get_all_costs(t_conf_pt)

        # Get destination configuration pionts whose last uav is in LoS with the UE
        if self._destination == "los":
            des_ind = self.get_conf_pt_inds_with_los_to_ue(t_conf_pt, ue_loc)
        elif self._destination == "nearest":
            des_ind = self.find_nearest_conf_pt_ind(
                t_conf_pt[:, -1, :][:, None, :], ue_loc[None, :])
        elif self._destination == "min_ue_rate":
            des_ind = self.get_conf_pt_inds_with_at_least_min_ue_rate(
                t_conf_pt, bs_loc, ue_loc)
        else:
            raise ValueError

        l_shortest_path_inds = self.get_shortest_path(m_cost,
                                                      ind_node_start=0,
                                                      ind_nodes_end=des_ind)

        if l_shortest_path_inds is not None:
            l_shortest_path = [t_conf_pt[ind] for ind in l_shortest_path_inds]
            return l_shortest_path
        else:
            # This may happen if there are corners that make that adjacent grid
            # points are found not to be connected by self.get_all_costs().

            print('RR cannot find the shortest path returns None')
            return None

    def resample(self, lm_path_coarse, samp_int, max_uav_speed):

        if lm_path_coarse is None:
            return None

        # a list of time traveling from one conf_pt to another.
        l_time_arrive_at_conf_pt = [0]

        for ind_waypt in range(1, len(lm_path_coarse)):

            l_time_arrive_at_conf_pt.append(
                l_time_arrive_at_conf_pt[-1] + np.max(
                    np.linalg.norm(lm_path_coarse[ind_waypt] -
                                   lm_path_coarse[ind_waypt - 1],
                                   axis=1)) / max_uav_speed)

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

        return lm_path_resampled

    def get_conf_pts(self, bs_loc, ue_loc, mode="random"):
        """ Returns a num_conf_pts x 2 x 3 array. All the configuration points
        must be in Qfree"""
        if mode == "random":
            return self.get_conf_pts_random()
        elif mode == "inter":
            return self.get_conf_pts_random_at_inter()
        elif mode == "grid":
            return self.get_conf_pts_on_grid(bs_loc)
        elif mode == "core":
            return self.get_conf_pts_from_core(bs_loc, ue_loc)
        elif mode == "feasible":
            return self.get_conf_pts_from_feasible_path(bs_loc, ue_loc)

    # submodules for get_conf_pts

    def get_conf_pts_random(self):

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

            if self.is_in_Qfree(m_conf_pt):
                l_conf_pt.append(m_conf_pt)

        return np.array(l_conf_pt)

    def get_conf_pts_random_at_inter(self):
        def find_road_intersections():
            """Finds the coordinates of the intersections of the block limits and the x and y axes.

            Returns a (num_inter_y_axis x num_inter_x_axis x 2) tensor contains the coordinates of the intersections.
            """
            def find_road_coor_1d(block_limits, grid_max, safety_margin=5):
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

            v_x_coor_inter = find_road_coor_1d(env.block_limits_x,
                                               env.slf.grid.max_x)

            v_y_coor_inter = find_road_coor_1d(env.block_limits_y,
                                               env.slf.grid.max_y)

            num_inter_x = v_x_coor_inter.shape[0]
            num_inter_y = v_y_coor_inter.shape[0]

            m_x_coor = np.tile(v_x_coor_inter.reshape(1, num_inter_x),
                               (num_inter_y, 1))
            m_y_coor = np.tile(v_y_coor_inter.reshape(num_inter_y, 1),
                               (1, num_inter_x))

            return np.stack((m_x_coor, m_y_coor), axis=-1)

        l_conf_pts = []

        env = self._environment

        # Discretize the configuration space
        t_inter_pts = find_road_intersections()

        def draw_a_conf_pt_at_intersections(t_inter_pts):

            num_loc_x = t_inter_pts.shape[1]
            num_loc_y = t_inter_pts.shape[0]

            m_available_pts = np.ones((num_loc_x, num_loc_y), dtype=bool)

            l_uav_ind = []

            # Append the 1st UAV
            ind_x = np.random.randint(0, num_loc_x)
            ind_y = np.random.randint(0, num_loc_y)
            m_available_pts[ind_x, ind_y] = False  # make as not available
            l_uav_ind.append([ind_x, ind_y])

            # Draw the indices of the coordinates of other UAVs
            # Iteratively draw
            l_choice = ['row', 'col']
            while len(l_uav_ind) < self._num_uavs:

                previous_uav = l_uav_ind[-1]

                if random.choice(l_choice) == 'row':
                    # The next uav will have the same row as the previous one
                    if np.mod(previous_uav[0], 2) == 0:
                        ind_x = previous_uav[0] + np.random.randint(0, 2)
                    else:
                        ind_x = previous_uav[0] - np.random.randint(0, 2)

                    ind_y = np.random.randint(0, num_loc_y)
                else:
                    # The next uav will have the same column as the previous one
                    ind_x = np.random.randint(0, num_loc_x)

                    if np.mod(previous_uav[1], 2) == 0:
                        ind_y = previous_uav[1] + np.random.randint(0, 2)
                    else:
                        ind_y = previous_uav[1] - np.random.randint(0, 2)

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

            l_conf_pts.append(draw_a_conf_pt_at_intersections(t_inter_pts))

        return np.array(l_conf_pts)

    def get_conf_pts_on_grid(self, bs_loc):

        fly_grid = self._environment.fly_grid
        m_grid_pts = fly_grid.list_pts()
        num_grid_pts = m_grid_pts.shape[0]

        def draw_a_conf_pt_in_los(bs_loc):
            """
                TODO: fix. 
                
                Returns a configuration point whose 1st UAV is in LoS with bs_loc
            
            """
            l_available = num_grid_pts * [True]
            # draw the 1st uav
            uav1_ind, l_available = self.draw_a_uav_ind_in_los(
                m_grid_pts, bs_loc, l_available)
            l_uav_ind = [uav1_ind]

            for ind_uav in range(1, self._num_uavs):

                pt2_ind, l_available = self.draw_a_uav_ind_in_los(
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
                        if np.linalg.norm(np.ravel(conf_pt -
                                                   l_conf_pts[ind])) == 0:
                            l_conf_pts.pop()
                            break

        return np.array(l_conf_pts) + np.random.rand(len(l_conf_pts),
                                                     self._num_uavs, 3)

    def get_conf_pts_from_core(self, bs_loc, ue_loc):
        """
        This function produces a set of configuration points for 2 UAVs where:

        The locations of UAV2 are drawn uniformly at random from the shortest
        path between bs_loc and the destination. The shortest path goes through
        points that have LoS with at least one point that has LoS with the BS. 

        The locations of UAV1 are drawn from the locations with LOS with the BS
        with a probabability that is proportional to the number of points in the
        aforementioned shortest path that are in LOS with that location.
        
        """

        l_pts_in_los_with_bs = self.get_pts_in_los_with_bs(bs_loc)

        assert len(l_pts_in_los_with_bs) != 0

        # Plan trajectory of UAV2 along points in LOS with at least a point in `l_pts_in_los_with_bs`
        m_shortest_path_uav_2 = np.array(
            self.
            _get_shortest_path_uav_2_thru_pts_with_los_with_pts_with_los_with_bs(
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
                    l_inds_uav_1_in_los_with_uav_2[ind_pt_uav_2].append(
                        ind_pt_uav_1)
                    num_possible_qpts += 1

        # Draw conf pts
        l_inds_locs_uav_1 = []
        l_inds_locs_uav_2 = []
        v_num_remaining_uav_1_pts = np.array(
            [len(l) for l in l_inds_uav_1_in_los_with_uav_2])

        for _ in range(np.minimum(self._num_nodes, num_possible_qpts)):

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

            ind_loc_uav_1 = np.random.choice(v_inds_los_locs_uav_1, p=v_prob)
            l_inds_locs_uav_1.append(ind_loc_uav_1)

            l_inds_uav_1_in_los_with_uav_2[ind_loc_uav_2].remove(ind_loc_uav_1)
            v_num_remaining_uav_1_pts[ind_loc_uav_2] -= 1

        m_locs_uav_2 = np.array(m_shortest_path_uav_2)[l_inds_locs_uav_2]
        m_locs_uav_1 = np.array(l_pts_in_los_with_bs)[l_inds_locs_uav_1]

        t_conf_pts = np.concatenate(
            (m_locs_uav_1[:, None, :], m_locs_uav_2[:, None, :]), axis=1)

        return np.array(t_conf_pts)

    def get_conf_pts_from_feasible_path(self, bs_loc, ue_loc):
        def sample_around_feas_path(lm_feas_path):
            def sample_conf_pts_around(conf_pt, num_pts):
                pos_uav_1, pos_uav_2 = conf_pt

                def sample_pt_inds_around_uav_pos(pos_uav, l_feas_pt_inds):
                    def get_dist(pt_1, pt_2):
                        dist = np.linalg.norm(pt_1 - pt_2)
                        if dist == 0:
                            return np.Inf
                        else:
                            return dist

                    l_dist_to_uav = [
                        get_dist(pos_uav, t_all_pts[ind_pt])
                        for ind_pt in l_feas_pt_inds
                    ]
                    v_probs = 1 / np.array(l_dist_to_uav)
                    v_probs = v_probs / np.sum(v_probs)
                    while True:
                        yield np.random.choice(l_feas_pt_inds, p=v_probs)

                g_pt_inds_uav_1 = sample_pt_inds_around_uav_pos(
                    pos_uav_1, l_feas_pt_inds_uav_1)
                g_pt_inds_uav_2 = sample_pt_inds_around_uav_pos(
                    pos_uav_2, l_feas_pt_inds_uav_2)

                lm_conf_pts = []
                for ind_pt_uav_1, ind_pt_uav_2 in zip(g_pt_inds_uav_1,
                                                      g_pt_inds_uav_2):

                    m_conf_pt = np.concatenate((t_all_pts[None, ind_pt_uav_1],
                                                t_all_pts[None, ind_pt_uav_2]),
                                               axis=0)

                    if len(lm_conf_pts) >= num_pts:
                        return lm_conf_pts

                    if self.is_in_Qfree(m_conf_pt):
                        lm_conf_pts.append(m_conf_pt)

            t_all_pts = self._environment.fly_grid.list_pts()

            l_feas_pt_inds_uav_1 = self.grid_pt_inds_fea_uav_1(bs_loc)

            l_feas_pts_uav_1 = [t_all_pts[ind] for ind in l_feas_pt_inds_uav_1]

            l_feas_pt_inds_uav_2 = [
                ind for ind in range(len(t_all_pts))
                if self.are_in_los(t_all_pts[ind], l_feas_pts_uav_1)
            ]

            lm_conf_pts = copy.deepcopy(lm_feas_path)

            for ind, conf_pt in enumerate(lm_feas_path):
                lm_conf_pts += sample_conf_pts_around(
                    conf_pt,
                    num_pts=np.floor(self._num_nodes / len(lm_feas_path)))

            return lm_conf_pts

        lm_feas_path = self.get_feasible_trajectory(bs_loc, ue_loc)
        if lm_feas_path is None:
            return None

        return np.array(sample_around_feas_path(lm_feas_path))

    # submodules for get_conf_pts_from_feasible_path

    def get_feasible_trajectory(self, bs_loc, ue_loc):
        """"
            Returns: list of 2 x 3 conf. pts with a feasible trajectory for both
            UAVs if it exists. If it does not exist, it returns None. 
        
        """
        def get_path_of_both_uavs_given_uav_2_path(l_initial_path_uav_2):
            """
            given the path of UAV2, it finds a suitable path for UAV1 so that
            UAV1 is all the time in LOS with both UAV2 and the BS. 

            To this end, this function may stop UAV2 at some of its waypoints to
            give time to UAV1 to move to the required position. 

            If no feasible UAV1 path exists, it returns None. 
            
            """
            def get_m_cost_extended_graph():

                # Total number of nodes
                num_nodes = 0
                for l_inds in ll_uav1_fea_inds_at_each_uav2_wpts:
                    num_nodes += len(l_inds)

                m_costs = np.inf * np.ones((num_nodes, num_nodes))

                def get_block(l_inds_1, l_inds_2):
                    """
                    It returns a matrix whose [i,j]-th entry is 1 if l_inds_1[i] and
                    l_inds_2[j] are adjacent and np.Inf otherwise.

                    TODO: optimize to compute only the lower triangular part.
                    """
                    def get_cost_for_inds(ind_1, ind_2):
                        if ind_1 == ind_2:
                            return 0.
                        out = fly_grid.are_adjacent(m_grid_pts[ind_1],
                                                    m_grid_pts[ind_2])
                        if out:
                            return np.linalg.norm(m_grid_pts[ind_1] -
                                                  m_grid_pts[ind_2])
                        else:
                            return np.Inf

                    m_block = np.array([[
                        get_cost_for_inds(ind_1, ind_2) for ind_2 in l_inds_2
                    ] for ind_1 in l_inds_1])

                    return m_block

                ind_block_start_row = 0

                for ind_block in range(
                        num_blocks):  #Iterates along block rows of m_costs

                    ind_block_end_row = ind_block_start_row + len(
                        ll_uav1_fea_inds_at_each_uav2_wpts[ind_block])

                    # Block diagonal
                    m_costs[ind_block_start_row:ind_block_end_row,
                            ind_block_start_row:ind_block_end_row] = get_block(
                                ll_uav1_fea_inds_at_each_uav2_wpts[ind_block],
                                ll_uav1_fea_inds_at_each_uav2_wpts[ind_block])

                    if ind_block < num_blocks - 1:
                        # Block diagonal right above the main block diagonal

                        ind_block_end_next_row = ind_block_end_row + len(
                            ll_uav1_fea_inds_at_each_uav2_wpts[ind_block + 1])

                        m_costs[
                            ind_block_start_row:ind_block_end_row,
                            ind_block_end_row:
                            ind_block_end_next_row] = get_block(
                                ll_uav1_fea_inds_at_each_uav2_wpts[ind_block],
                                ll_uav1_fea_inds_at_each_uav2_wpts[ind_block +
                                                                   1])

                    ind_block_start_row = ind_block_end_row

                return m_costs

            def eind_to_block_and_fly_grid_pt_ind(eind):
                ind_block = np.where(
                    eind - v_eind_first_enode_each_block < 0)[0][0] - 1
                ind_within_block = eind - v_eind_first_enode_each_block[
                    ind_block]
                ind_fly_grid_pt = ll_uav1_fea_inds_at_each_uav2_wpts[
                    ind_block][ind_within_block]
                return ind_block, ind_fly_grid_pt

            def block_and_fly_grid_pt_inds_to_eind(ind_block, ind_fly_grid_pt):
                if ind_block > 0:
                    raise NotImplemented
                return np.where(ll_uav1_fea_inds_at_each_uav2_wpts[ind_block]
                                == ind_fly_grid_pt)[0][0]

            def get_ind_nodes_end():
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
                    ind_block, fly_grid_pt_ind = eind_to_block_and_fly_grid_pt_ind(
                        eind)
                    pt_uav_1 = m_grid_pts[fly_grid_pt_ind]
                    pt_uav_2 = l_initial_path_uav_2[ind_block][0]
                    conf_pt = np.concatenate(
                        (pt_uav_1[None, :], pt_uav_2[None, :]), axis=0)
                    return PathPlanner.conf_pt_to_is_ue_rate_gt_min(
                        self._channel,
                        conf_pt,
                        min_uav_rate=self.min_uav_rate,
                        min_ue_rate=self.min_ue_rate,
                        bs_loc=bs_loc,
                        ue_loc=ue_loc,
                        v_los=v_los)

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

            fly_grid = self._environment.fly_grid
            m_grid_pts = fly_grid.list_pts()

            # We construct an extended graph. Each node of the extended graph
            # (enode) corresponds to a fly grid pt index and a block index.

            if self.mode_connect == 'min_rate_only':
                s_grid_pts_inds_at_least_rate_with_bs = set(
                    self.get_grid_pt_inds_at_least_rate(bs_loc,
                                                        2 * self.min_uav_rate,
                                                        fly_grid=fly_grid))
                ll_uav1_fea_inds_at_each_uav2_wpts = [
                    list(
                        s_grid_pts_inds_at_least_rate_with_bs.intersection(
                            set(
                                self.get_grid_pt_inds_at_least_rate(
                                    pt, self.min_uav_rate,
                                    fly_grid=fly_grid))))
                    for pt in l_initial_path_uav_2
                ]

            else:
                # 0.  Find the list of lists of intersections between the set of fly grid points in LoS with the BS and the sets of fly grid points in LoS with waypoints of the trajectory of uav 2.
                s_grid_pt_inds_fea_uav_1 = set(
                    self.grid_pt_inds_fea_uav_1(bs_loc))

                ll_uav1_fea_inds_at_each_uav2_wpts = [
                    list(
                        s_grid_pt_inds_fea_uav_1.intersection(
                            set(
                                self.
                                grid_pt_inds_in_los_n_at_least_min_uav_rate_with(
                                    pt[0])))) for pt in l_initial_path_uav_2
                ]

            num_blocks = len(ll_uav1_fea_inds_at_each_uav2_wpts)

            l_num_nodes_each_block = [
                len(l_inds) for l_inds in ll_uav1_fea_inds_at_each_uav2_wpts
            ]

            # The i-th entry of the following list is the extended node index (eind)
            # of the first enode in the i-th block. The last entry is the total
            # number of enodes.
            v_eind_first_enode_each_block = np.cumsum([0] +
                                                      l_num_nodes_each_block)

            # 1. Construct m_cost
            m_cost = get_m_cost_extended_graph()

            # 2. Invoke shortest_path()
            # TODO: fix ind_node_start to ensure LOS with BS
            ind_enode_start = block_and_fly_grid_pt_inds_to_eind(
                0, fly_grid.nearest_ind(bs_loc))

            ind_nodes_end = get_ind_nodes_end()
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
                eind_to_block_and_fly_grid_pt_ind(eind)
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

        def lift_path(l_path):
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

        def concat_takeoff_and_landing(l_path_lifted):
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

        # UAV 2
        if self.mode_connect == 'min_rate_only':
            l_tentative_path_uav_2 = self._get_shortest_path_uav_2_thru_pts_greater_rate_with_pts_greater_rate_with_bs(
                bs_loc, ue_loc)
        else:
            l_tentative_path_uav_2 = self._get_shortest_path_uav_2_thru_pts_with_los_with_pts_with_los_with_bs(
                bs_loc, ue_loc)

        if l_tentative_path_uav_2 is None:
            # This happens when self.min_ue_rate is too high.
            return None

        initial_pos_uav_2, final_pos_uav_2 = l_tentative_path_uav_2[
            0], l_tentative_path_uav_2[-1]

        while True:
            # UAV 1
            lm_path = get_path_of_both_uavs_given_uav_2_path(
                concat_takeoff_and_landing(l_tentative_path_uav_2))

            if lm_path is not None:
                return lm_path

            print("lifting path")
            l_tentative_path_uav_2 = lift_path(l_tentative_path_uav_2)

            if l_tentative_path_uav_2 is None:
                raise ValueError(
                    "No path can be found even by lifting the path of UAV2. Most likely some buildings are higher than the highest grid point. "
                )

    def get_conf_pt_inds_with_los_to_ue(self, t_conf_pt, ue_loc):
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

    def get_conf_pt_inds_with_at_least_min_ue_rate(self, t_conf_pt, bs_loc,
                                                   ue_loc):
        """     

            Returns:
                l_des_ind: list of indices of configuration points for which the UE rate exceeds self._min_ue_rate.

        """

        l_des_ind = []
        for ind_conf_pt in range(t_conf_pt.shape[0]):
            rate = PathPlanner.conf_pt_to_ue_rate(self._channel,
                                                  t_conf_pt[ind_conf_pt],
                                                  self.min_uav_rate,
                                                  bs_loc=bs_loc,
                                                  ue_loc=ue_loc)
            if rate > self.min_ue_rate:
                l_des_ind.append(ind_conf_pt)

        return l_des_ind

    def get_pts_in_los_with_bs(self, bs_loc):
        fly_grid = self._environment.fly_grid
        m_grid_pts = fly_grid.list_pts()
        l_pts_in_los_with_bs = [
            m_grid_pts[ind] for ind in self.grid_pt_inds_in_los_with(bs_loc)
        ]

        return l_pts_in_los_with_bs

    def grid_pt_inds_fea_uav_1(self, bs_loc):
        fly_grid = self._environment.fly_grid
        m_grid_pts = fly_grid.list_pts()

        l_grid_pt_inds_in_los_with_bs = self.grid_pt_inds_in_los_with(bs_loc)
        l_grid_pt_inds_fea_uav_1 = []

        for ind in l_grid_pt_inds_in_los_with_bs:
            if self.compute_capacity(bs_loc,
                                     m_grid_pts[ind]) >= 2 * self.min_uav_rate:
                l_grid_pt_inds_fea_uav_1.append(ind)

        return l_grid_pt_inds_fea_uav_1

    def get_feasible_loc_uav_1(self, bs_loc):
        """"
            Return a list of points that satisfy the following conditions:
                + in LoS with BS,
                + capacity from BS >= 2 * min_uav_rate
        """

        fly_grid = self._environment.fly_grid
        m_grid_pts = fly_grid.list_pts()

        l_feasible_loc_uav_1 = [
            m_grid_pts[ind] for ind in self.grid_pt_inds_fea_uav_1(bs_loc)
        ]

        return l_feasible_loc_uav_1

    def is_uav_2_conf_pt(self, pt, l_feasible_pts_uav_1):
        """
            Returns true if pt can be a location for uav 2. This occurs when there exists A in l_feasible_pts_uav_1 s.t.
                + pt and A are in LoS,
                + capacity(A,pt) >= min_uav_rate
        """

        for loc_uav_1 in l_feasible_pts_uav_1:
            if self.are_in_los(pt, loc_uav_1) and self.compute_capacity(
                    pt, loc_uav_1) >= self.min_uav_rate:
                return True
        return False

    def grid_pt_inds_in_los_n_at_least_min_uav_rate_with(self, ref_pt):
        fly_grid = self._environment.fly_grid
        m_grid_pts = fly_grid.list_pts()

        l_grid_pt_inds_los_n_rate = []

        l_grid_pt_inds_in_los_with_ref_pt = self.grid_pt_inds_in_los_with(
            ref_pt)

        for ind in l_grid_pt_inds_in_los_with_ref_pt:
            if self.compute_capacity(ref_pt,
                                     m_grid_pts[ind]) >= self.min_uav_rate:
                l_grid_pt_inds_los_n_rate.append(ind)

        return l_grid_pt_inds_los_n_rate

    def _get_shortest_path_uav_2_thru_pts_with_los_with_pts_with_los_with_bs(
            self, bs_loc, ue_loc):
        """
        Returns:
            l_shortest_path: list of 1 x 3 arrays that contains the shortest
            trajectory of a UAV through points that are in LOS with points that
            are in LOS with `bs_loc`. 
        """
        def min_ue_rate_grid_pt_inds(m_grid_pts_uav_2, bs_loc, ue_loc):
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

            l_inds_los_with_ue = self.grid_pt_inds_in_los_with(
                ue_loc, grid=fly_grid_uav_2)

            return [
                ind_pt for ind_pt in l_inds_los_with_ue
                if can_get_min_ue_rate(m_grid_pts_uav_2[ind_pt])
            ]

        # Feasible positions for UAV1
        l_feasible_pts_uav_1 = self.get_feasible_loc_uav_1(bs_loc)

        # Feasible points for UAV2
        fly_grid_uav_2 = self._environment.fly_grid.clone()
        # disable points that are not in LoS with any point in
        # l_feasible_pts_uav_1
        fly_grid_uav_2.disable_by_indicator(
            lambda pt: not self.is_uav_2_conf_pt(pt, l_feasible_pts_uav_1))

        m_grid_pts_uav_2 = fly_grid_uav_2.list_pts()
        t_conf_pt = m_grid_pts_uav_2[:, None, :]

        ind_start = fly_grid_uav_2.nearest_ind(bs_loc)

        if self._destination == "los":
            l_ind_end = self.grid_pt_inds_in_los_with(ue_loc,
                                                      grid=fly_grid_uav_2)
        elif self._destination == "nearest":
            l_ind_end = fly_grid_uav_2.nearest_ind(ue_loc)
        elif self._destination == "min_ue_rate":
            l_ind_end = min_ue_rate_grid_pt_inds(m_grid_pts_uav_2, bs_loc,
                                                 ue_loc)
        else:
            raise ValueError

        if len(l_ind_end) == 0:
            print("No feasible trajectory of UAV2 found")
            return None

        m_cost = self.get_all_costs(t_conf_pt,
                                    grid=fly_grid_uav_2,
                                    neighbor_mode="grid")

        l_shortest_path_inds = self.get_shortest_path(m_cost,
                                                      ind_node_start=ind_start,
                                                      ind_nodes_end=l_ind_end)

        l_shortest_path = [t_conf_pt[ind] for ind in l_shortest_path_inds]

        return l_shortest_path

    def _get_shortest_path_uav_2_thru_pts_greater_rate_with_pts_greater_rate_with_bs(
            self, bs_loc, ue_loc):
        """
        Returns:
            l_shortest_path: list of 1 x 3 arrays that contains the shortest
            trajectory of a UAV through points that have at least (r_cc + min_ue_rate) with points that have at least (2r_cc + min_ue_rate) with `bs_loc`. 
        """
        fly_grid = self._environment.fly_grid

        def get_pts_at_least_rate_with_ref_pts(ref_pts,
                                               min_rate,
                                               fly_grid=None):

            assert fly_grid is not None

            m_grid_pts = fly_grid.list_pts()

            l_pts_at_least_rate_with_ref_pts = [
                m_grid_pts[ind] for ind in self.get_grid_pt_inds_at_least_rate(
                    ref_pts, min_rate, fly_grid=fly_grid)
            ]
            return l_pts_at_least_rate_with_ref_pts

        def des_min_ue_rate_grid_pt_inds(m_grid_pts_uav_2, bs_loc, ue_loc):
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

            l_inds_within_rate_with_ue = self.get_grid_pt_inds_at_least_rate(
                ue_loc, self.min_ue_rate, fly_grid=fly_grid_uav_2)

            return [
                ind_pt for ind_pt in l_inds_within_rate_with_ue
                if can_get_min_ue_rate(m_grid_pts_uav_2[ind_pt])
            ]

        # Feasible positions for UAV1
        l_pts_at_least_2rcc_with_bs = get_pts_at_least_rate_with_ref_pts(
            bs_loc, 2 * self.min_uav_rate, fly_grid=fly_grid)

        # Feasible points for UAV2
        fly_grid_uav_2 = self._environment.fly_grid.clone()
        # disable points that dont have at least min_uav_rate with any point in
        # l_pts_at_least_2rcc_with_bs
        fly_grid_uav_2.disable_by_indicator(
            lambda pt: not self.is_at_least_rate(
                pt, l_pts_at_least_2rcc_with_bs, self.min_uav_rate))

        m_grid_pts_uav_2 = fly_grid_uav_2.list_pts()
        t_conf_pt = m_grid_pts_uav_2[:, None, :]

        ind_start = fly_grid_uav_2.nearest_ind(bs_loc)

        if self._destination == "los":
            l_ind_end = self.grid_pt_inds_in_los_with(ue_loc,
                                                      grid=fly_grid_uav_2)
        elif self._destination == "nearest":
            l_ind_end = fly_grid_uav_2.nearest_ind(ue_loc)
        elif self._destination == "min_ue_rate":
            l_ind_end = des_min_ue_rate_grid_pt_inds(m_grid_pts_uav_2, bs_loc,
                                                     ue_loc)
        else:
            raise ValueError

        if len(l_ind_end) == 0:
            print("No feasible trajectory of UAV2 found")
            return None

        m_cost = self.get_all_costs(t_conf_pt,
                                    grid=fly_grid_uav_2,
                                    neighbor_mode="grid")

        l_shortest_path_inds = self.get_shortest_path(m_cost,
                                                      ind_node_start=ind_start,
                                                      ind_nodes_end=l_ind_end)

        l_shortest_path = [t_conf_pt[ind] for ind in l_shortest_path_inds]

        return l_shortest_path

    def draw_a_uav_ind_in_los(self, m_grid_pts, pt1_loc, l_available):
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

        while all(m_grid_pts[pt2_ind] == pt1_loc) or (l_available[pt2_ind]
                                                      == False):
            pt2_ind = np.random.randint(m_grid_pts.shape[0])

        if self.are_in_los(pt1_loc, m_grid_pts[pt2_ind]):
            l_available[pt2_ind] = False
            return pt2_ind, l_available
        else:
            return self.draw_a_uav_ind_in_los(m_grid_pts, pt1_loc, l_available)

    def is_in_Qfree(self, m_conf_pt):

        num_uavs = m_conf_pt.shape[0]
        if self.mode_connect == 'min_rate_only':

            for ind_uav in range(num_uavs):
                if ind_uav == 0:
                    current_rate = self.compute_capacity(
                        self.bs_loc, m_conf_pt[0])
                else:
                    current_rate = np.maximum(
                        0, current_rate -
                        (num_uavs - ind_uav) * self.min_uav_rate)
                if current_rate == 0:
                    return False
        else:
            for ind_uav in range(num_uavs - 1):

                if not self.are_in_los(self.bs_loc, m_conf_pt[ind_uav]):
                    return False

                if not self.are_in_los(m_conf_pt[ind_uav],
                                       m_conf_pt[ind_uav + 1]):
                    return False

                if self.compute_capacity(
                        self.bs_loc,
                        m_conf_pt[ind_uav]) < 2 * self.min_uav_rate:
                    return False

                if self.compute_capacity(
                        m_conf_pt[ind_uav],
                        m_conf_pt[ind_uav + 1]) < self.min_uav_rate:
                    return False
        return True

    def are_connected(self, m_conf_pt_1, m_conf_pt_2):
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
            if not self.is_in_Qfree(m_conf_pt_1 + ind_step *
                                    self._step_connectivity * m_direction):
                return False
        return True

    def get_all_costs(self,
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
                num_conf_pts = len(t_conf_pt)

                # l_target = [
                #     ind_conf_pt for ind_conf_pt in range(num_conf_pts)
                #     if grid.are_adjacent(m_conf_pt[0], t_conf_pts[ind_conf_pt,
                #                                                   0])
                # ]

                return self._environment.fly_grid.get_inds_adjacent(
                    m_conf_pt[0], t_conf_pts[:, 0, :])

        num_nodes = t_conf_pt.shape[0]

        # Determine if each conf pt is connected to its nearest neighbors
        m_cost = np.tile(np.inf, (num_nodes, num_nodes))

        for ind_conf_pt, v_conf_pt in enumerate(t_conf_pt):
            for ind_nn in get_inds_neighbors(v_conf_pt, t_conf_pt):
                if np.sum(m_cost[ind_conf_pt, :] < np.inf
                          ) >= self._max_num_neighbors:
                    break

                if m_cost[ind_conf_pt, ind_nn] < np.inf:
                    continue

                if self.are_connected(v_conf_pt, t_conf_pt[ind_nn]):
                    m_cost[ind_conf_pt,
                           ind_nn] = self.get_cost(v_conf_pt,
                                                   t_conf_pt[ind_nn])
                    m_cost[ind_nn, ind_conf_pt] = m_cost[ind_conf_pt, ind_nn]

            if np.all(m_cost[ind_conf_pt, :] == np.inf):
                print(
                    f"Node {ind_conf_pt} is not connected with any other node."
                )

        return m_cost

    def get_cost(self, pt1, pt2):
        return np.max(np.sqrt(np.sum((pt1 - pt2)**2, axis=1)))

    def get_shortest_path(self, m_cost, ind_node_start, ind_nodes_end):
        """
        Args:

            - m_cost is an num_nodes x num_nodes matrix whose (m,n)-th entry is
              the cost of going from node m to node n. It is np.Inf if m and n
              are not connected. 

            - ind_nodes_end: set/list of indices of the possible destinations.
              If `ind_nodes_end` is an integer, then it is understood as
              {ind_nodes_end}.           
                
        Returns:

        l_path: list of indices of nodes forming the shortest path where 

            - l_path[0] = ind_node_start
            - l_path[-1] = ind_node_end
            - sum_i m_cost[ l_path[i] , l_path[i+1] ] is minimum

            l_path is None if there is no path between the nodes with indices
            ind_node_start and ind_node_end. 

        If ind_nodes_end is empty, the function returns None.

        """

        num_nodes = m_cost.shape[0]

        if (type(ind_nodes_end) == int) or (type(ind_nodes_end) == np.int64):
            ind_nodes_end = {ind_nodes_end}
        else:  # else, it is a set or a list
            ind_nodes_end = set(ind_nodes_end)
        if len(ind_nodes_end) == 0:
            return None

        # Find the node with minimum distance to the source and the node
        # immediately before it in the shortest path.
        def f_node_to_add(l_distances, l_nodes_added_so_far):
            """ 
                The i-th entry of l_distances contains the length of the best
                path known so far between the starting node and the i-th node. 

                Among the nodes not yet included in the shortest path, find 

                    1. the node with the minimum distance so far to the
                        starting node 
                    
                    2. and the node in the shortest path that the newly added
                       node should go through to go towards the starting point.
            """

            # Initialize minimum distance for next node
            min_distance = np.Inf

            # Find the node that has not been added yet with minimum distance to
            # the starting node.
            for ind_node in range(num_nodes):
                if l_distances[
                        ind_node] < min_distance and l_nodes_added_so_far[
                            ind_node] == False:
                    min_distance = l_distances[ind_node]
                    node_to_add = ind_node

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
                l_candidate_prev_nodes = []
                for ind_node in range(num_nodes):
                    if m_cost[ind_node,
                              node_to_add] < np.inf and l_nodes_added_so_far[
                                  ind_node] == True:
                        # why condition after and? --> bc. any node N that has
                        # not been added has a greater distance to the starting
                        # point than `node_to_add` because of the way
                        # `node_to_add` was chosen and by the fact that to reach
                        # N, one must necessarily go through one added node and
                        # (possibly) one neighbor of an added node.
                        l_candidate_prev_nodes.append(ind_node)

                # Choose the previous node in l_candidate_prev_nodes that
                # results in the shortest path to `node_to_add`.
                l_dist_to_source = []
                for ind_node in l_candidate_prev_nodes:
                    l_dist_to_source.append(m_cost[ind_node, node_to_add] +
                                            l_distances[ind_node])

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
            l_distances = [
                np.Inf
            ] * num_nodes  #l_distances = [sys.maxsize] * num_nodes
            l_distances[ind_node_start] = 0

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
                    l_distances, l_nodes_added_so_far)

                l_nodes_added_so_far[node_to_add] = True
                l_nodes_to_go[node_to_add] = node_to_go

                # Stop if the shortest path to one of the nodes in
                # `set_end_nodes` has been found.
                if node_to_add in set_end_nodes:
                    break

                # Update distances from the source to all nodes that
                #
                #       + are not in l_nodes_added_so_far, AND
                #
                #       + are adjacent to node_to_add (the newly added node)
                #
                # if their distances to the starting node become smaller when
                # going through node_to_add.
                for node_compared in range(num_nodes):
                    if m_cost[node_to_add,node_compared] < np.inf and l_nodes_added_so_far[node_compared] == False and \
                        l_distances[node_compared] > l_distances[node_to_add] + m_cost[node_to_add,node_compared]:

                        l_distances[node_compared] = l_distances[
                            node_to_add] + m_cost[node_to_add, node_compared]

                # The following condition holds iff there are no adjacent
                # nodes to the nodes added so far and there are still nodes
                # that have not been added. Due to the stopping condition
                # above (break), node `ind_node_end` has not been added.
                # Thus, there is no path towards that node.
                if (not all(l_nodes_added_so_far) == True) and (
                        list(np.array(l_distances) < sys.maxsize)
                        == l_nodes_added_so_far):
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

    @staticmethod
    def find_nearest_conf_pt_ind(t_conf_pt, conf_pt):
        return np.argmin(
            np.sum(np.sum((t_conf_pt - conf_pt[None, :, :])**2, axis=1),
                   axis=1))

    def conf_to_work_space_pt(conf_pt):
        work_pt = [
            [conf_pt[0], conf_pt[1], conf_pt[2]],
            [conf_pt[3], conf_pt[4], conf_pt[5]],
        ]
        return work_pt

    def grid_pt_inds_in_los_with(self, ref_pts, grid=None):
        """
        Args:

        ref_pt is either a 3D point or a list/np.array of 3D points.
        
        Returns:

        list of indices of the fly grid points that are in LOS with at least one
        point in `ref_pts`.
        
        """
        if grid is None:
            grid = self._environment.fly_grid

        fvf = FunctionVectorField(
            grid=grid, fun=lambda pt: np.array([self.are_in_los(pt, ref_pts)]))

        return list(np.where(fvf.list_vals()[:, 0])[0])

    def get_grid_pt_inds_at_least_rate(self, ref_pts, min_rate, fly_grid=None):
        """
        Args:

        ref_pts is either a 3D point or a list/np.array of 3D points.
        
        Returns:

        list of indices of the fly grid points that have at least min_rate with at least one point in `ref_pts`.
        
        """

        assert fly_grid is not None

        fvf = FunctionVectorField(
            grid=fly_grid,
            fun=lambda pt: np.array(
                [self.is_at_least_rate(pt, ref_pts, min_rate)]))

        return list(np.where(fvf.list_vals()[:, 0])[0])
