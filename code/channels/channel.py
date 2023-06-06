import abc
from IPython.core.debugger import set_trace
import numpy as np
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from gsim.utils import startTimer, printTimer
from gsim.gfigure import GFigure

from common.fields import VectorField, FunctionVectorField

from common.utilities import dB_to_natural, natural_to_dB, dbm_to_watt, watt_to_dbm

import logging

log = logging.getLogger("channel")

speed_of_light = 3e8

log.debug("inside this place...")


# Base class for channels. It models a free-space channel.
class Channel():

    def __init__(
        self,
        freq_carrier=3e9,
        antenna_dbgain_tx=0,
        antenna_dbgain_rx=0,
        tx_dbpower=0,
        noise_dbpower=0,
        bandwidth=1,
        min_link_capacity=None,
        max_link_capacity=None,
        disable_gridpts_by_dominated_verticals=True,
        #max_uav_total_rate=None,
    ):
        """ Args:

            `tx_dbpower`: used just for computing capacity maps.

            `min_link_capacity`: if not None, sets to 0 the rates at all links
            whose capacity is below this value.

            `max_link_capacity`: if not None, sets to `max_link_capacity` the
            rates at all links whose capacity is above this value.            

            If `disable_gridpts_by_dominated_verticals` is True, then certain
            grid points of `map.grid` are disabled when generating a map; see
            `Channel.capacity_map`.

            If you want to be able to change the parameters above after an
            object has been constructed, extend the mechanism by which the map
            cache is handled. 


        """
        # remove
        # The grid is necessary to use geometric quantities such as distance
        # between points. However, a subclass may provide the gain or response
        # for arbitrary pairs of points. Thus, the grid here need not have the
        # same geometry as the grid used outside of this module.
        #assert map_base_grid
        #self.map_base_grid = map_base_grid

        self.wavelength = speed_of_light / freq_carrier

        self.antenna_dbgain_tx = antenna_dbgain_tx
        self.antenna_dbgain_rx = antenna_dbgain_rx
        self.tx_dbpower = tx_dbpower
        self.noise_dbpower = noise_dbpower
        self.bandwidth = bandwidth
        self.max_link_capacity = max_link_capacity
        self.min_link_capacity = min_link_capacity
        self.disable_gridpts_by_dominated_verticals = disable_gridpts_by_dominated_verticals

        self._capacity_map_cache = dict()

    # Abstract methods
    def dbgain(self, pt_1, pt_2):
        """Returns the gain in dB between positions `pt_1` 
        and `pt_2`. """
        raise NotImplementedError

    def dbgain_free_space(self, pt_1, pt_2):
        """Returns the gain in dB between positions `pt_1` 
        and `pt_2`"""
        if (pt_1.ndim > 1) or (pt_2.ndim > 1):
            raise NotImplementedError
        dist = np.linalg.norm(pt_1 - pt_2)

        return self.dist_to_dbgain_free_space(
            dist,
            wavelength=self.wavelength,
            antenna_dbgain_tx=self.antenna_dbgain_tx,
            antenna_dbgain_rx=self.antenna_dbgain_rx,
        )

    @staticmethod
    def dist_to_dbgain_free_space(distance,
                                  wavelength=0.1,
                                  antenna_dbgain_tx=0,
                                  antenna_dbgain_rx=0):
        # Ensure far field
        distance = max(distance, wavelength)
        return antenna_dbgain_tx + antenna_dbgain_rx + natural_to_dB(
            (wavelength / (4 * np.pi * distance))**2)

    @staticmethod
    def dbgain_to_dist_free_space(dbgain,
                                  wavelength=0.1,
                                  antenna_dbgain_tx=0,
                                  antenna_dbgain_rx=0):
        return 1 / ((4 * np.pi / wavelength) * np.sqrt(
            dB_to_natural(dbgain - antenna_dbgain_tx - antenna_dbgain_rx)))

    def dbgain_from_pt(self, grid=None, pt_1=None):
        """Returns a VectorField that provides the gain between point
        `pt_1` and all grid points.

        """
        assert pt_1 is not None
        assert grid is not None

        def dbgain(pt_2):
            return [self.dbgain(pt_1, pt_2)]

        return FunctionVectorField(grid=grid, fun=dbgain)

    def capacity_from_pt(self, grid, pt_1, debug=0):
        """Returns a VectorField that provides the gain between point
        `pt_1` and all grid points.

        """
        assert pt_1 is not None

        def capacity(pt_2):
            # snr = dB_to_natural(self.tx_dbpower + self.dbgain(pt_1, pt_2) -
            #                     self.noise_dbpower)
            # C = np.log2(1 + snr)
            if debug:
                print("pt_1: ", pt_1)
                print("pt_2: ", pt_2)
            C = self.dbgain_to_capacity(self.dbgain(pt_1, pt_2))
            if (self.min_link_capacity
                    is not None) and (C < self.min_link_capacity):
                C = 0
            return [C]

        return FunctionVectorField(grid=grid, fun=capacity)

    def dbgain_to_capacity(self, dbgain):
        snr = dB_to_natural(self.tx_dbpower + dbgain - self.noise_dbpower)
        C = self.bandwidth * np.log2(1 + snr)
        return C

    def capacity_to_dbgain(self, capacity):
        min_snr = 2**(capacity / self.bandwidth) - 1
        min_rx_dbpower = natural_to_dB(min_snr) + self.noise_dbpower
        min_dbgain = min_rx_dbpower - self.tx_dbpower
        return min_dbgain

    def max_distance_for_rate(self, min_rate):
        """Returns the maximum distance between tx and rx so that the capacity
        in free space equals `min_rate`.  """
        # min_snr = 2**min_rate - 1
        # min_rx_dbpower = natural_to_dB(min_snr) + self.noise_dbpower
        # min_dbgain = min_rx_dbpower - self.tx_dbpower
        min_dbgain = self.capacity_to_dbgain(min_rate)

        radius = self.dbgain_to_dist_free_space(
            min_dbgain,
            wavelength=self.wavelength,
            antenna_dbgain_tx=self.antenna_dbgain_tx,
            antenna_dbgain_rx=self.antenna_dbgain_rx,
        )

        return radius

    def max_ground_radius_for_height(self, min_rate, height):
        max_dist = self.max_distance_for_rate(min_rate=min_rate)
        return np.sqrt(max_dist**2 - height**2)

    # Maps
    def capacity_maps(self, grid, user_coords):
        """ Obtains one capacity map per user. 
        
        Args: 
        
        - `user_coords` is num_users x 3
        
        Returns:
        - list of VectorFields, each one corresponding to a user.

        """

        return [
            self.capacity_from_pt(grid=grid, pt_1=uc) for uc in user_coords
        ]

    def capacity_map(
        self,
        grid,
        user_coords,
        #clip_by_max_link_capacity=True,
    ):
        """         
        Args: 

        - `user_coords` is num_users x 3

        Returns:
        - `map`: VectorField with an entry per user.

        If  `self.max_link_capacity` is not None, the entries greater than
        self.max_link_capacity are set to self.max_link_capacity.

        If `self.disable_gridpts_by_dominated_verticals` is True, then certain
        grid points of `map.grid` are disabled; see
        VectorField.disable_gridpts_by_dominated_verticals


        For caching to work properly, `grid` should not be modified between
        calls to this function. 

        """

        def in_cache():
            d = self._capacity_map_cache
            if 'map' not in d.keys():
                return False

            # A map is in the cache. Let us check if it is the same as requested
            return  (np.all(d['user_coords'] == np.array(user_coords))) \
                and (d['max_link_capacity'] == self.max_link_capacity)\
                and (d['disable_gridpts_by_dominated_verticals'] == self.disable_gridpts_by_dominated_verticals)\
                and (d['grid_id'] == id(grid))

        def compute_map():
            map = VectorField.concatenate(self.capacity_maps(
                grid, user_coords))
            if self.max_link_capacity is not None:
                map.clip(upper=self.max_link_capacity)
            if self.disable_gridpts_by_dominated_verticals:
                map = map.clone()  # so that we do not modify `grid`
                map.disable_gridpts_by_dominated_verticals()
            return map

        if not in_cache():
            # Map computation
            startTimer("Capacity map computation")

            map = compute_map()
            printTimer("Capacity map computation")

            # Update cache
            self._capacity_map_cache.update({
                'map':
                map,
                'user_coords':
                np.array(user_coords),
                'max_link_capacity':
                self.max_link_capacity,
                'disable_gridpts_by_dominated_verticals':
                self.disable_gridpts_by_dominated_verticals,
                'grid_id':
                id(grid)
            })

        return self._capacity_map_cache['map']

    # Metrics for UAV placements
    def assess_placement(
            self,
            grid,
            uav_coords=None,
            user_coords=None,
            #min_user_rate=None,
            max_uav_total_rate=None):
        """
        See rate_allocation.
        """
        if uav_coords is None:
            assessment = {
                "num_uavs": None,
                "user_rates_mult": None,
                "min_user_rate_mult": None,
                "sum_rate_mult": None,
                "user_rates_sing": None,
                "min_user_rate_sing": None,
                "sum_rate_sing": None
            }
            return assessment

        # Obtain max. rates for each pair (uav, user)
        # The following is num_users x num_points.
        map = self.capacity_map(grid=grid, user_coords=user_coords)
        m_capacity_all_gridpts = map.list_vals().T
        m_capacity_uavs = m_capacity_all_gridpts[:,
                                                 map.grid.
                                                 nearest_inds(uav_coords)]

        assessment = dict()
        assessment["num_uavs"] = len(uav_coords)
        #assessment["m_capacity_uavs="] = m_capacity_uavs

        # Each MU can connect to multiple UAVs
        R_opt = self.rate_allocation(m_capacity_uavs,
                                     max_uav_total_rate,
                                     mode="multiple")
        user_rates = np.sum(R_opt, axis=1)
        assessment["user_rates_mult"] = user_rates
        assessment["min_user_rate_mult"] = np.min(user_rates)
        assessment["sum_rate_mult"] = np.sum(user_rates)

        # Each MU can connect to only one UAV
        R_opt = self.rate_allocation(m_capacity_uavs,
                                     max_uav_total_rate,
                                     mode="single")
        user_rates = np.sum(R_opt, axis=1)
        assessment["user_rates_sing"] = user_rates
        assessment["min_user_rate_sing"] = np.min(user_rates)
        assessment["sum_rate_sing"] = np.sum(user_rates)

        return assessment
        # log.debug("UAV rates=")
        # log.debug(np.sum(R_opt, axis=0))

    def rate_allocation(
        self,
        m_capacity_uavs,
        max_uav_total_rate=None,
        mode="multiple",
    ):
        """Returns a matrix of the same shape as m_capacity_uavs with the rates
        for each pair (user,uav).

            `mode` can be:

                -  "single": each user associates only with one UAV. The 
                  rates of all other links to that user are set to 0. 

                -  "multiple": in this case a user can be associated with multiple UAVs. 

                    If `max_uav_total_rate` is not None, the allocation is such that the 
                    minimum user rate is maximized. 

                    If `max_uav_total_rate` is None, all UAVs tx at the maximum rate.

        See notes_cartography.pdf 2021/07/22.
        """

        def single_allocation():

            m_capacity_out = np.zeros(shape=m_capacity_uavs.shape)
            ind_max = np.argmax(m_capacity_uavs, axis=1)
            m_capacity_out[range(num_users),
                           ind_max] = m_capacity_uavs[range(num_users),
                                                      ind_max]

            inds = [(row, ind_max[row]) for row in range(len(m_capacity_out))]
            return m_capacity_out

        num_users, num_uavs = m_capacity_uavs.shape

        if mode == "single":
            return single_allocation()

        if max_uav_total_rate is None:
            # No limits -> just tx the maximum
            return m_capacity_uavs

        c = np.zeros((num_users * num_uavs + 1, ))
        c[-1] = -1

        Aub_top = np.concatenate(
            (-np.tile(np.eye(num_users), reps=(1, num_uavs)),
             np.ones((num_users, 1))),
            axis=1)
        Aub_bottom = np.concatenate((np.repeat(
            np.eye(num_uavs), num_users, axis=1), np.zeros((num_uavs, 1))),
                                    axis=1)
        Aub = np.concatenate((Aub_top, Aub_bottom), axis=0)

        bub = np.concatenate((np.zeros(
            (num_users, )), max_uav_total_rate * np.ones((num_uavs))),
                             axis=0)

        bounds_top = np.concatenate((np.zeros(
            (num_users * num_uavs, 1)), np.ravel(m_capacity_uavs.T)[:, None]),
                                    axis=1)
        bounds_bottom = np.array([[0, None]])
        bounds = np.concatenate((bounds_top, bounds_bottom), axis=0)

        res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds)

        x_opt = res.x
        r_opt = x_opt[:num_users * num_uavs]

        R_opt = np.reshape(r_opt, (num_uavs, num_users)).T

        log.debug("min rate = ", -res['fun'])
        return R_opt

    def feasible_placement_exists(self,
                                  grid,
                                  user_coords,
                                  min_user_rate,
                                  max_uav_total_rate=None):

        if max_uav_total_rate is not None:
            raise NotImplementedError

        if min_user_rate is None:
            return True

        m_capacity = self.capacity_map(grid=grid,
                                       user_coords=user_coords).list_vals().T

        s = np.sum(m_capacity, axis=1)
        coverable_users = (s >= min_user_rate)
        return np.sum(coverable_users) == len(s)


class FreeSpaceChannel(Channel):

    def dbgain(self, pt_1, pt_2):
        return self.dbgain_free_space(pt_1, pt_2)