import abc
import copy
from placement.placers import FlyGrid
from common.grid import RectangularGrid3D
from IPython.core.debugger import set_trace
import numpy as np
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from gsim_conf import use_mayavi
if use_mayavi:
    print("Loading MayaVi...")
    from mayavi import mlab
    print("done")

from gsim.gfigure import GFigure

from common.fields import FunctionVectorField
from common.utilities import natural_to_dB, dbm_to_watt, watt_to_dbm, nearest_row
from channels.channel import Channel


class Building():
    def __init__(self,
                 sw_corner=None,
                 ne_corner=None,
                 height=None,
                 absorption=1):
        assert sw_corner[2] == 0
        assert ne_corner[2] == 0
        assert ne_corner[0] > sw_corner[0]
        assert ne_corner[1] > sw_corner[1]
        self.sw_corner = sw_corner
        self.ne_corner = ne_corner
        self.height = height
        self.absorption = absorption  # dB/m

    @property
    def nw_corner(self):
        return np.array([self.sw_corner[0], self.ne_corner[1], 0])

    @property
    def se_corner(self):
        return np.array([self.ne_corner[0], self.sw_corner[1], 0])

    def plot(self):
        """Adds a rectangle per side of the building to the current figure.  """
        assert use_mayavi

        def high(pt):
            return np.array([pt[0], pt[1], self.height])

        def lateral_face(pt1, pt2):
            return np.array([
                [high(pt1), high(pt2)],
                [pt1, pt2],
            ])

        def plot_face(face):
            mlab.mesh(face[..., 0], face[..., 1], face[..., 2])
            mlab.mesh(face[..., 0],
                      face[..., 1],
                      face[..., 2],
                      representation='wireframe',
                      color=(0, 0, 0))

        # top face
        face = np.array([
            [high(self.nw_corner), high(self.ne_corner)],
            [high(self.sw_corner), high(self.se_corner)],
        ])
        plot_face(face)

        # west face
        face = lateral_face(self.nw_corner, self.sw_corner)
        plot_face(face)

        # north face
        face = lateral_face(self.nw_corner, self.ne_corner)
        plot_face(face)

        # east face
        face = lateral_face(self.ne_corner, self.se_corner)
        plot_face(face)

        # south face
        face = lateral_face(self.sw_corner, self.se_corner)
        plot_face(face)

        # face = np.array([
        #     [self.nw_corner, self.ne_corner],
        #     [self.sw_corner, self.se_corner],
        # ])
        # plot_face(face)

    @property
    def min_x(self):
        return self.sw_corner[0]

    @property
    def max_x(self):
        return self.ne_corner[0]

    @property
    def min_y(self):
        return self.sw_corner[1]

    @property
    def max_y(self):
        return self.ne_corner[1]

    # True for points inside the building, False otherwise
    def indicator(self, coords):
        x, y, z = coords
        return self.absorption * ((x >= self.min_x) & (x <= self.max_x) &
                                  (y >= self.min_y) & (y <= self.max_y) &
                                  (z < self.height))


class SLField(FunctionVectorField):
    pass


class Environment():
    area_len = None
    _fly_grid = None

    def __init__(self, area_len=None):
        if area_len:
            self.area_len = area_len

    @property
    def fly_grid(self):
        return self._fly_grid


class UrbanEnvironment(Environment):
    l_users = None  # list of points corresponding to the user locations
    l_lines = None  # list of lists of points. A line is plotted for each list

    # `dl_uavs`: dict whose keys are strings and whose values are lists of
    # points corresponding to the UAV locations. Each list will be represented
    # with a different color and marker.
    dl_uavs = dict()

    def __init__(self,
                 base_fly_grid=None,
                 buildings=None,
                 num_pts_slf_grid=None,
                 **kwargs):
        """ Args: 

            `base_fly_grid`: object of class FlyGrid. The gridpoints inside
            buildings are disabled. The resulting grid can be accessed through
            `self.fly_grid`.

            `num_pts_slf_grid`: vector with 3 entries corresponding to the
            number of points along the X, Y, and Z dimensions
        """

        super().__init__(**kwargs)

        # self.grid = slf_grid
        self.buildings = buildings
        slf_grid = RectangularGrid3D(num_pts=num_pts_slf_grid,
                                     area_len=self.area_len)
        self.slf = SLField(grid=slf_grid, fun=self.f_buildings_agg)
        if base_fly_grid is not None:
            self._fly_grid = base_fly_grid
            self._fly_grid.disable_by_indicator(self.building_indicator)

    def building_indicator(self, coords):
        """Each entry is the indicator for a building scaled by the
        absorption. """
        x, y, z = coords
        return [building.indicator(coords) for building in self.buildings]

    # Combine all possible buildings into a single SLF.
    @property
    def f_buildings_agg(self):
        return lambda coords: np.max(self.building_indicator(coords))[None, ...
                                                                      ]

    def random_pts_on_street(self, num_pts):
        """ Returns a `num_pts` x 3 matrix whose rows contain the coordinates of
        points drawn uniformly at random on the ground and out of the buildings.

        The limits of the buildings are given by the function self.f_buildings.
        The building boundaries obtained in this way may not coincide with the
        voxel boundaries.
        """

        f_indicator = self.f_buildings_agg

        def filter_street(l_pts):
            return np.array([pt for pt in l_pts if not f_indicator(pt)])

        num_remaining = num_pts
        l_pts = []
        while num_remaining > 0:
            # TODO: generate the points using self.area_len rather than self.slf.grid
            new_pts = self.slf.grid.random_pts(num_pts, z_val=0)
            new_pts = filter_street(new_pts)
            if len(new_pts) > 0:
                l_pts += [new_pts]
            num_remaining -= len(new_pts)

        user_coords = np.concatenate(l_pts, axis=0)[:num_pts, :]

        # plt.plot(user_coords[:,0], user_coords[:,1], ".")
        # plt.show()
        return user_coords

    def disable_flying_gridpts_by_dominated_verticals(self, channel):
        """Disables gridpoints of the fly grid according to the positions of the users"""

        assert self.l_users is not None

        channel = copy.deepcopy(channel)
        channel.disable_gridpts_by_dominated_verticals = True

        map = channel.capacity_map(grid=self.fly_grid,
                                   user_coords=self.l_users)

        self._fly_grid.t_enabled = map.grid.t_enabled

    def plot_buildings(self):
        if not use_mayavi:
            return self.slf.plot_as_blocks()
        else:
            for building in self.buildings:
                building.plot()

    def plot(self,
             bgcolor=(1., 1., 1.),
             fgcolor=(0., 0., 0.),
             azim=None,
             elev=None):
        if not use_mayavi:
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            ax = self.plot_buildings()
            self._plot_pt_dl(self.dl_uavs, ax=ax)
            self._plot_pts(self.l_users, style="o", ax=ax)
            if self._fly_grid is not None:
                self._plot_pts(self._fly_grid.list_pts(),
                               style=".",
                               ax=ax,
                               color=(0, .7, .9),
                               alpha=0.3)
            self._plot_lines(ax)

            ax.view_init(azim=azim, elev=elev)
            return ax

        else:
            mlab.figure(bgcolor=bgcolor, fgcolor=bgcolor)
            self.plot_buildings()
            self._plot_pt_dl(self.dl_uavs, scale_factor=self.area_len[0] / 20.)
            self._plot_pts(self.l_users,
                           style="2dsquare",
                           scale_factor=3 * self.area_len[0] / 150.,
                           color=(1, 1, 1))
            if self._fly_grid is not None:
                self._plot_pts(self._fly_grid.list_pts(),
                               style="cube",
                               scale_factor=self.area_len[0] / 150.,
                               color=(0, 0.7, 0.9),
                               opacity=.3)
            self._plot_lines()
            self._plot_ground()
            return

    def show(self):
        if not use_mayavi:
            plt.show()
        else:
            mlab.show()

    def _plot_ground(self):
        min_x = 0  # g.min_x
        min_y = 0  # g.min_y
        max_x, max_y, max_z = self.area_len
        ground = np.array([
            [[0, max_y, 0], [max_x, max_y, 0]],
            [[0, 0, 0], [max_x, 0, 0]],
        ])
        mlab.mesh(ground[..., 0],
                  ground[..., 1],
                  ground[..., 2],
                  color=(0, 0, 0))
        ax = mlab.axes(extent=[min_x, max_x, min_y, max_y, 0, max_z],
                       nb_labels=4)
        # ax.axes.font_factor = .8

    def _plot_pt_dl(self, dl_pts, **kwargs):
        """`dl_pts` is a dict of lists of points."""
        if len(dl_pts) == 0:
            return
        if use_mayavi:
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
            ][:len(dl_pts)]
        else:
            l_markers = ['x', 'o', 'd'][:len(dl_pts)]
        # m_colors = np.random.random((len(dl_pts), 3))
        l_colors = [[.4, .8, .0], [.7, 0, 0], [.7, .3, .1], [.8, .0, .1],
                    [.9, .5, .0]][0:len(dl_pts)]
        print("Legend:")
        for name, marker, color in zip(dl_pts.keys(), l_markers, l_colors):
            self._plot_pts(dl_pts[name],
                           style=marker,
                           color=tuple(color),
                           **kwargs)
            print(f"{marker} --> {name}")

    def _plot_pts(self,
                  l_pts,
                  style,
                  ax=None,
                  scale_factor=3.,
                  color=(.5, .5, .5),
                  **kwargs):
        if l_pts is not None:
            pts = np.array(l_pts)
            if not use_mayavi:
                ax.plot(pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        style,
                        color=color,
                        **kwargs)
            else:
                mlab.points3d(pts[:, 0],
                              pts[:, 1],
                              pts[:, 2],
                              mode=style,
                              scale_factor=scale_factor,
                              color=color,
                              **kwargs)

    def _plot_lines(self, ax=None):
        if self.l_lines is not None:
            for line in self.l_lines:
                pts = np.array(line)
                if not use_mayavi:
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2])
                else:
                    mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2])


class BlockUrbanEnvironment(UrbanEnvironment):
    def __init__(self,
                 num_pts_fly_grid=[5, 5, 3],
                 min_fly_height=20,
                 building_height=20,
                 building_absorption=1,
                 **kwargs):
        """ Args: 

            `num_pts_slf_grid` and `num_pts_fly_grid`: vectors with 3 entries
            corresponding to the number of points along the X, Y, and Z
            dimensions

            `building_height`: if None, it is generated as
            min(max(np.random.normal(loc=15, scale=15), 5), 80). If it is a
            vector of length 2, the height of each building is generated as a
            uniform distribution in the interval indicated by the entries of
            this vector. 
        """
        assert "area_len" not in kwargs.keys()
        assert "base_fly_grid" not in kwargs.keys()
        base_fly_grid = FlyGrid(area_len=self.area_len,
                                num_pts=num_pts_fly_grid,
                                min_height=min_fly_height)

        super().__init__(base_fly_grid=base_fly_grid,
                         buildings=self._get_buildings(
                             height=building_height,
                             absorption=building_absorption),
                         **kwargs)

    def _get_buildings(self, height, absorption):
        def get_height(height):
            if height is None:
                h = np.random.normal(loc=15, scale=15)
                return min(max(h, 5), 80)
            height = np.array(height)
            if height.ndim == 0:
                return height
            elif height.ndim == 1:
                return np.random.uniform(low=height[0], high=height[1])
            else:
                raise ValueError

        l_buildings = []
        for block_x in self.block_limits_x:
            for block_y in self.block_limits_y:
                bld = Building(sw_corner=[block_x[0], block_y[0], 0],
                               ne_corner=[block_x[1], block_y[1], 0],
                               height=get_height(height),
                               absorption=absorption)
                l_buildings.append(bld)

        return l_buildings


class BlockUrbanEnvironment1(BlockUrbanEnvironment):
    area_len = [100, 80, 50]
    block_limits_x = np.array([[20, 30], [50, 60], [80, 90]])
    block_limits_y = np.array([[15, 30], [50, 60]])


class BlockUrbanEnvironment2(BlockUrbanEnvironment):
    area_len = [1000, 1000, 100]

    street_width = 100
    building_width = 120

    def __init__(self, *args, **kwargs):
        def block_limits(len_axis):
            block_ends = np.arange(self.street_width + self.building_width,
                                   self.area_len[0],
                                   step=self.street_width +
                                   self.building_width)[:, None]

            return np.concatenate(
                [np.maximum(block_ends - self.building_width, 0), block_ends],
                axis=1)

        self.block_limits_x = block_limits(self.area_len[0])
        self.block_limits_y = block_limits(self.area_len[1])

        super().__init__(*args, **kwargs)


class GridBasedBlockUrbanEnvironment(BlockUrbanEnvironment):
    height_over_min_enabled_height = 3

    def __init__(self,
                 area_len=[100, 80, 50],
                 num_pts_fly_grid=[5, 5, 3],
                 min_fly_height=20,
                 building_absorption=1,
                 building_height=None,
                 **kwargs):
        """ 
            The z=0 plane is divided in pixels whose centers are given by the
            flight gridpoints. Along each dimension, one every two pixels are
            occupied by a building. 
                
            Args: 

            `num_pts_slf_grid` and `num_pts_fly_grid`: vectors with 3 entries
            corresponding to the number of points along the X, Y, and Z
            dimensions

            `building_height`: if None, it is set to
            `self.height_over_min_enabled_height` units above the min enabled
            flying height.
        """
        assert "base_fly_grid" not in kwargs.keys()
        assert num_pts_fly_grid[0] % 2 == 1
        assert num_pts_fly_grid[1] % 2 == 1
        base_fly_grid = FlyGrid(area_len=area_len,
                                num_pts=num_pts_fly_grid,
                                min_height=min_fly_height)

        if building_height is None:
            building_height = base_fly_grid.min_enabled_height + self.height_over_min_enabled_height

        def edges_to_block_limits(edges):
            if len(edges) % 2 == 1:
                edges = edges[0:-1]
            return np.reshape(edges, (-1, 2))

        self.block_limits_x = edges_to_block_limits(base_fly_grid.t_edges[0, 0,
                                                                          0,
                                                                          1:])
        self.block_limits_y = edges_to_block_limits(
            np.flip(base_fly_grid.t_edges[1, 0, :-1, 0]))

        super(BlockUrbanEnvironment, self).__init__(
            area_len=area_len,
            base_fly_grid=base_fly_grid,
            buildings=self._get_buildings(height=building_height,
                                          absorption=building_absorption),
            **kwargs)


class RandomHeightGridBasedBlockUrbanEnvironment(BlockUrbanEnvironment):
    def __init__(self,
                 area_len=[100, 100, 70],
                 num_pts_fly_grid=[5, 5, 3],
                 min_fly_height=20,
                 building_absorption=1,
                 building_height=None,
                 **kwargs):

        base_fly_grid = FlyGrid(area_len=area_len,
                                num_pts=num_pts_fly_grid,
                                min_height=min_fly_height)

        def edges_to_block_limits(edges):
            if len(edges) % 2 == 1:
                edges = edges[0:-1]
            return np.reshape(edges, (-1, 2))

        self.block_limits_x = edges_to_block_limits(base_fly_grid.t_edges[0, 0,
                                                                          0,
                                                                          1:])
        self.block_limits_y = edges_to_block_limits(
            np.flip(base_fly_grid.t_edges[1, 0, :-1, 0]))

        super(BlockUrbanEnvironment, self).__init__(
            area_len=area_len,
            base_fly_grid=base_fly_grid,
            buildings=self._get_buildings(height=building_height,
                                          absorption=building_absorption),
            **kwargs)


class RayTracingEnvironment(Environment):
    d_metadata = {
        'ottawa_02': {
            'threshold_ch_gain': -200,  # if the channel gain value at the i-th user
            # location is less than `self.threshold_ch_gain` then
            # the i-th user location is inside the building.
            'v_height_of_fly_grid_points': [40, 60, 80],  # height of fly grid point slabs

            'num_pts': np.array([5, 7, 5]),  # num_fly_grid_pts_x, num_fly_grid_pts_y, num_slab_z,
            # where num_slab_z = max(v_height_of_fly_grid_points)/spacing_between_slab + 1
            'file_path_ch_gain': lambda \
                    ind_fly_grid, ind_fly_grid_height:
            f"./data/Ottawa_02/ch_gains/ch_gain_height_{ind_fly_grid_height}_fly_grid_{ind_fly_grid + 1}.txt",

            'file_path_fly_grid_point_locations': lambda
                ind_fly_grid_height: f"./data/Ottawa_02/fly_grid_{ind_fly_grid_height}_height_locations.txt",

            "discription":
                "This data was generated using   "
                "the city Ottawa with Wireless Insite"
                "ray-tracing software with 6 fly gird points at the height of 40m."
        },

        'ottawa_03': {
            'threshold_ch_gain': -200,  # if the channel gain value at the i-th user
            # location is less than `self.threshold_ch_gain` then
            # the i-th user location is inside the building.
            'v_height_of_fly_grid_points': [40, 60, 80],  # height of fly grid point slabs

            'num_pts': np.array([5, 7, 5]),  # num_fly_grid_pts_x, num_fly_grid_pts_y, num_slab_z,
            # where num_slab_z = max(v_height_of_fly_grid_points)/spacing_between_slab + 1
            'file_path_fs_loss': lambda \
                    ind_fly_grid, ind_fly_grid_height:
            f"./data/Ottawa_03/free_space_path_loss/fs_loss_height_{ind_fly_grid_height}_fly_grid_{ind_fly_grid + 1}.txt",

            'file_path_fly_grid_point_locations': lambda
                ind_fly_grid_height: f"./data/Ottawa_02/fly_grid_{ind_fly_grid_height}_height_locations.txt",

            "discription":
                "This data was generated using   "
                "the city Ottawa with Wireless Insite"
                "ray-tracing software with 3 fly gird slabs at the height of"
                "40m, 60m, and 80m, where each slab has 35 fly grid points"
                "i.e. total 105 fly grid points."
        },

        'ottawa_04': {
            'threshold_ch_gain': -200,  # if the channel gain value at the i-th user
            # location is less than `self.threshold_ch_gain` then
            # the i-th user location is inside the building.
            'v_height_of_fly_grid_points': [1040, 1060, 1080],  # height of fly grid point slabs

            'num_pts': np.array([5, 7, 55]),  # num_fly_grid_pts_x, num_fly_grid_pts_y, num_slab_z,
            # where num_slab_z = max(v_height_of_fly_grid_points)/spacing_between_slab + 1
            'file_path_ch_gain': lambda \
                    ind_fly_grid, ind_fly_grid_height:
            f"./data/Ottawa_04/ch_gains/ch_gain_height_{ind_fly_grid_height}_fly_grid_{ind_fly_grid + 1}.txt",

            'file_path_fly_grid_point_locations': lambda
                ind_fly_grid_height: f"./data/Ottawa_04/fly_grid_{ind_fly_grid_height}_height_locations.txt",

            "discription":
                "This data was generated using   "
                "the city Ottawa with Wireless Insite"
                "ray-tracing software with 3 fly gird slabs at the height of"
                "1040m, 1060m, and 1080m, where each slab has 35 fly grid points"
                "i.e. total 105 fly grid points. Antenna is dipole."
        },

        'ottawa_05': {
            'threshold_ch_gain': -200,  # if the channel gain value at the i-th user
            # location is less than `self.threshold_ch_gain` then
            # the i-th user location is inside the building.
            'v_height_of_fly_grid_points': [1040, 1060, 1080],  # height of fly grid point slabs

            'num_pts': np.array([5, 7, 55]),  # num_fly_grid_pts_x, num_fly_grid_pts_y, num_slab_z,
            # where num_slab_z = max(v_height_of_fly_grid_points)/spacing_between_slab + 1
            'file_path_ch_gain': lambda \
                    ind_fly_grid, ind_fly_grid_height:
            f"./data/Ottawa_05/ch_gains/ch_gain_height_{ind_fly_grid_height}_fly_grid_{ind_fly_grid + 1}.txt",

            'file_path_fly_grid_point_locations': lambda
                ind_fly_grid_height: f"./data/Ottawa_05/fly_grid_{ind_fly_grid_height}_height_locations.txt",

            "discription":
                "This data was generated using   "
                "the city Ottawa with Wireless Insite"
                "ray-tracing software with 3 fly gird slabs at the height of"
                "1040m, 1060m, and 1080m, where each slab has 35 fly grid points"
                "i.e. total 105 fly grid points. This is a dataset with free space channel gain"
                "without any reflections. The antenna used in this case is an"
                "isotropic antenna."
        },

        'ottawa_06': {
            'threshold_ch_gain': -120,  # if the channel gain value at the i-th user
            # location is less than `self.threshold_ch_gain` then
            # the i-th user location is inside the building.
            'v_height_of_fly_grid_points': [40, 60, 80],  # height of fly grid point slabs

            'num_pts': np.array([5, 7, 5]),  # num_fly_grid_pts_x, num_fly_grid_pts_y, num_slab_z,
            # where num_slab_z = max(v_height_of_fly_grid_points)/spacing_between_slab + 1
            'file_path_ch_gain': lambda \
                    ind_fly_grid, ind_fly_grid_height:
            f"./data/Ottawa_06/ch_gains/ch_gain_height_{ind_fly_grid_height}_fly_grid_{ind_fly_grid + 1}.txt",

            'file_path_fly_grid_point_locations': lambda
                ind_fly_grid_height: f"./data/Ottawa_06/fly_grid_{ind_fly_grid_height}_height_locations.txt",

            "discription":
                "This data was generated using   "
                "the city Ottawa with Wireless Insite"
                "ray-tracing software with 3 fly gird slabs at the height of"
                "40m, 60m, and 80m, where each slab has 35 fly grid points"
                "i.e. total 105 fly grid points. This is a dataset generated "
                "with 6 reflections and 1 diffraction. The antennas used in this case are an"
                "isotropic antenna."
        },

        'ottawa_01': {
            'threshold_ch_gain': -200,  # if the channel gain value at the i-th user
            # location is less than `self.threshold_ch_gain` then
            # the i-th user location is inside the building.
            'v_height_of_fly_grid_points': [50],  # height of fly grid point slabs

            'num_pts': np.array([2, 3, 2]),  # num_pts_x, num_pts_y, num_slab_z,
            # where num_slab_z = max(v_height_of_fly_grid_points)/spacing_between_slab + 1
            'file_path_ch_gain': lambda \
                    ind_fly_grid, ind_fly_grid_height:
            f"./data/Ottawa_01/Ottawa_ch_gain_fly_grid_point_{ind_fly_grid + 1}.txt",

            'file_path_fly_grid_point_locations': lambda
                ind_fly_grid_height: f"./data/Ottawa_01/Ottawa_tx_locations.txt",

            "discription":
                "This data was generated using   "
                "the city Ottawa with Wireless Insite"
                "ray-tracing software with 6 fly gird points at the height of 40m."
        },

    }
    l_users = None  # list of points corresponding to the user locations

    def __init__(self, *args, dataset=None, **kwargs):

        self._m_user_locs_outside_building_cache = None

        if dataset in self.d_metadata.keys():

            self.metadata = self.d_metadata[dataset]

            # if the channel gain value at the i-th user location is less than
            # `self.threshold_ch_gain` then
            # the i-th user location is inside the building.
            self.threshold_ch_gain = self.metadata[
                'threshold_ch_gain']  # in dB
            num_pts = self.metadata['num_pts']
            min_height = min(self.metadata['v_height_of_fly_grid_points'])

        else:
            raise ValueError(f'The dataset {dataset} does not exist. ')

        # read the data from the files
        self.m_user_locs, self.m_fly_grid_locs, self.m_ch_gains = self._read_data_from_txt(
        )

        # translate the coords such that minimum [x, y] = [0, 0]
        v_min_coords = np.min(self.m_fly_grid_locs, axis=0)
        v_min_coords[-1] = 0  # not to translate vertically
        self.m_user_locs -= v_min_coords
        self.m_fly_grid_locs -= v_min_coords

        # get the side length of the fly grid along x, y, z axes
        v_max_coords = list(np.max(self.m_fly_grid_locs, axis=0))

        def f_disable_indicator(coords):
            """This function returns True if the fly grid point with coordinates
            `coords` is inside a building. """

            return np.all(
                self.ch_dbgain_to_all_user_pts(coords) < self.threshold_ch_gain
            )

        if (num_pts <= 1).any():
            raise ValueError
        v_area_len = v_max_coords * num_pts / (num_pts - 1)

        self._fly_grid = FlyGrid(area_len=v_area_len,
                                 num_pts=num_pts,
                                 min_height=min_height,
                                 f_disable_indicator=f_disable_indicator)

        # debug
        def check_fly_grid():
            all_fly_grid_pts = self._fly_grid.list_pts(
                exclude_disabled_pts=False)
            all_fly_grid_pts_above_min_height = all_fly_grid_pts[
                all_fly_grid_pts[:, 2] >= min_height]

            for v_fly_grid_loc in self.m_fly_grid_locs:
                if v_fly_grid_loc not in all_fly_grid_pts_above_min_height:
                    print(
                        f'Point {v_fly_grid_loc} is in self.m_fly_grid_locs but not in all_fly_grid_pts_above_min_height'
                    )
            for v_fly_grid_loc in all_fly_grid_pts_above_min_height:
                if v_fly_grid_loc not in self.m_fly_grid_locs:
                    print(
                        f'Point {v_fly_grid_loc} is in all_fly_grid_pts_above_min_height but not in self.m_fly_grid_locs'
                    )

        check_fly_grid()

    def _read_data_from_txt(self):
        """

        Returns:
            m_user_locs: num_users x 3 matrix whose i-th entry contains
            the 3D coords of the i-th user

            m_fly_grid_locs: num_fly_grid_locs x 3 matrix whose i-th
            entry contains the 3D coords of the i-th fly grid point

            m_ch_gains: num_fly_grid_locs x num_users matrix whose
            (i,j)-th entry is the channel gain between the
                i-th fly grid point and the j-th user.
        """
        m_fly_grid_locs = None
        m_user_locs = None
        l_ch_gain_for_all_fly_grid = []

        for ind_fly_grid_height in self.metadata[
                'v_height_of_fly_grid_points']:
            file_path_tx_locations = self.metadata[
                'file_path_fly_grid_point_locations'](ind_fly_grid_height)

            df_fly_grid_locs = pd.read_csv(file_path_tx_locations,
                                           skiprows=3,
                                           index_col=False,
                                           header=None,
                                           sep=" ")

            if m_fly_grid_locs is None:
                m_fly_grid_locs = df_fly_grid_locs.iloc[:, 1:4].to_numpy()
            else:
                m_fly_grid_locs = np.vstack(
                    (m_fly_grid_locs, df_fly_grid_locs.iloc[:,
                                                            1:4].to_numpy()))

            num_fly_grid_locs = len(df_fly_grid_locs.iloc[:, 0])

            for ind_fly_grid in range(num_fly_grid_locs):
                if 'file_path_fs_loss' in self.metadata.keys():
                    file_path = self.metadata['file_path_fs_loss'](
                        ind_fly_grid, ind_fly_grid_height)
                    b_invert_to_get_gain = True
                else:
                    file_path = self.metadata['file_path_ch_gain'](
                        ind_fly_grid, ind_fly_grid_height)
                    b_invert_to_get_gain = False

                df_ch_gain_and_locs = pd.read_csv(file_path,
                                                  skiprows=3,
                                                  index_col=False,
                                                  header=None,
                                                  sep=" ")
                if b_invert_to_get_gain:
                    # if the file contains path loss take negative to change
                    # the values into channel gain.
                    v_ch_gain = -1 * df_ch_gain_and_locs.iloc[:, -1].to_numpy()
                else:
                    v_ch_gain = df_ch_gain_and_locs.iloc[:, -1].to_numpy()

                l_ch_gain_for_all_fly_grid.append(v_ch_gain)

                if m_user_locs is None:
                    # num_users x 3
                    m_user_locs = df_ch_gain_and_locs.iloc[:, 1:4].to_numpy()
                # l_user_locs.append(m_user_loc)

        # this is num_fly_grid_locs x num_users
        m_ch_gains = np.array(l_ch_gain_for_all_fly_grid)

        return m_user_locs, m_fly_grid_locs, m_ch_gains

    def random_pts_on_street(self, num_pts):
        """ Returns a `num_pts` x 3 matrix whose rows contain the coordinates of
        points drawn uniformly at random without replacement on the user
        location grid. 

        """

        if self._m_user_locs_outside_building_cache is None:
            self._m_user_locs_outside_building_cache = self.m_user_locs[
                np.where(
                    np.any(self.m_ch_gains >= self.threshold_ch_gain,
                           axis=0))[0]]

        m_random_pts_on_street = self._m_user_locs_outside_building_cache[
            np.random.choice(np.arange(
                0, len(self._m_user_locs_outside_building_cache)),
                             size=num_pts,
                             replace=False)]

        return m_random_pts_on_street

    def coords_to_ind_fly_grid_pt(self, coords):

        # return np.argmin(np.sum((self.m_fly_grid_locs - coords)**2, axis=1))
        return nearest_row(self.m_fly_grid_locs, coords)

    def coords_to_ind_user_pt(self, coords):

        # return np.argmin(np.sum((self.m_user_locs - coords)**2, axis=1))
        return nearest_row(self.m_user_locs, coords)

    def ch_dbgain(self, fly_grid_pt, user_pt):

        ind_fly_grid_pt = self.coords_to_ind_fly_grid_pt(fly_grid_pt)
        ind_user_pt = self.coords_to_ind_user_pt(user_pt)

        if (ind_fly_grid_pt is None) or (ind_user_pt is None):
            return -np.inf
        else:
            return self.m_ch_gains[ind_fly_grid_pt, ind_user_pt]

    def ch_dbgain_to_all_user_pts(self, fly_grid_pt):

        ind_fly_grid_pt = self.coords_to_ind_fly_grid_pt(fly_grid_pt)

        if (ind_fly_grid_pt is None):
            num_user_pts = self.m_ch_gains.shape[1]
            return np.tile(-np.inf, (num_user_pts, ))
        else:
            return self.m_ch_gains[ind_fly_grid_pt, :]
