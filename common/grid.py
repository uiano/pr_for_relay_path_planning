import copy
from IPython.core.debugger import set_trace
from common.utilities import project_to_interval, mat_argmin, mat_argmax, \
    natural_to_dB, dB_to_natural
import numpy as np
import matplotlib.pyplot as plt
from gsim.gfigure import GFigure
from common.utilities import empty_array

# import grid_utilities
try:
    import grid_utilities
except ModuleNotFoundError:
    print('ERROR: please install grid_utilities as described on README.md')
# import sys to test
import sys


class Grid():

    @staticmethod
    def plot_pts_3D(ax, coords=None, style="x"):
        """General utility while we do not extend GFigure.

            `ax` is a 3D axis

            `coords` is num_pts x 3

        """
        coords = np.array(coords)
        assert coords.ndim == 2
        ax.plot(np.array(coords[:, 0]), np.array(coords[:, 1]),
                np.array(coords[:, 2]), style)


class RectangularGrid3D(Grid):
    """External code should depend on the internal representation of the
        grid as least as possible.

    Internally, the grid can be thought of as a `num_pts_z x
    num_pts_y x num_pts_x` tensor `points` whose entries are 3D
    points. The coordinates of the point at the [i,j,k]-th entry of
    the tensor are

    x = gridpt_spacing_x * k
    y = gridpt_spacing_y * (num_pts_y - 1 - j)
    z = gridpt_spacing_z * i

    The origin corresponds therefore to the entry
    `[0,num_pts_y-1,0]`.

    With the above assignment, `points[i]` is a horizontal slice of
    the grid whose height is `gridpt_spacing_z * i`. Each of these
    horizontal slices is, therefore, a matrix of 3D points. The
    bottom-left corner of this matrix corresponds, therefore, to
    x=y=0.

    """

    # `t_enabled` is None or a Boolean tensor of shape num_pts_z x num_pts_y x
    # num_pts_x. It is used to list only enabled points. If None, all points are
    # assumed enabled.
    t_enabled = None

    def __init__(self, *args, area_len=None, num_pts=None, **kwargs):
        """Args:

        - `area_len`: list/vector of 3 positive scalars. If it is a scalar, it
          is broadcast to a vector. The entries are interpreted as [len_x,
          len_y, len_z]

        - `num_pts`: list/vector of 3 positive scalars. If it is a scalar, it is
          broadcast to a vector. The entries are interpreted as [num_pts_x,
          num_pts_y, num_pts_z].

        """

        # Input check: check that mandatory arguments were provided
        assert area_len is not None
        assert num_pts is not None

        super().__init__(*args, **kwargs)

        def broadcast(vec):
            if not hasattr(vec, "__len__"):
                return np.repeat([vec], 3)
            assert len(vec) == 3
            return np.array(vec)

        area_len = broadcast(area_len)
        num_pts = broadcast(num_pts)

        self.num_pts_x = num_pts[0]
        self.num_pts_y = num_pts[1]
        self.num_pts_z = num_pts[2]

        # spacing = voxel side length
        self.spacing = area_len / num_pts

    def clone(self):
        return copy.deepcopy(self)

    @property
    def num_pts(self):
        return self.num_pts_x * self.num_pts_y * self.num_pts_z

    @property
    def num_enabled_pts(self):
        if self.t_enabled is None:
            return self.num_pts
        else:
            return np.sum(self.t_enabled.astype(int))

    @property
    def min_x(self):
        return -self.spacing[0] / 2

    @property
    def max_x(self):
        return (self.num_pts_x - 1) * self.spacing[0] + self.spacing[0] / 2

    @property
    def min_y(self):
        return -self.spacing[1] / 2

    @property
    def max_y(self):
        return (self.num_pts_y - 1) * self.spacing[1] + self.spacing[1] / 2

    @property
    def min_z(self):
        return -self.spacing[2] / 2

    @property
    def max_z(self):
        return (self.num_pts_z - 1) * self.spacing[2] + self.spacing[2] / 2

    def is_within_limits(self, pt):
        """ Returns true if `pt` is within the limits of the grid."""
        return all([
            pt[0] >= self.min_x,
            pt[0] <= self.max_x,
            pt[1] >= self.min_y,
            pt[1] <= self.max_y,
            pt[2] >= self.min_z,
            pt[2] <= self.max_z,
        ])

    @property
    def origin(self):
        return np.zeros((3, ))

    @property
    def max_herror(self):
        """Maximum error in the horizontal plane after projecting a point onto
        the grid when all grid points are enabled."""
        return 1 / 2 * np.sqrt(self.spacing[0]**2 + self.spacing[1]**2)

    @property
    def min_enabled_height(self):
        """Returns the height of the enabled grid point wigh lowest height"""

        pts = self.list_pts(exclude_disabled_pts=True)
        return np.min(pts[:, 2])

    def nearest_z_ind(self, zval):
        """Returns the index of the z-slice whose z-value is closest to `val`. """
        assert zval is not None
        return int(
            np.min([
                np.max([np.round(zval / self.spacing[2]), 0]),
                self.num_pts_z - 1
            ]))

    def nearest_y_ind(self, yval):
        """Returns the index of the Y-slice whose y-value is closest to `val`. """
        assert yval is not None
        return self.num_pts_y - 1 - int(
            np.min([
                np.max([np.round(yval / self.spacing[1]), 0]),
                self.num_pts_y - 1
            ]))

    def nearest_z(self, zval):
        """Returns the z-coordinate in the grid that lies closest to `zval`. """
        return self.z_axis[self.nearest_z_ind(zval)]

    def nearest_pt(self, pt, exclude_disabled=True):
        return self.nearest_pt_ind(pt,
                                   exclude_disabled_pts=exclude_disabled)[0]

    def nearest_ind(self, pt, exclude_disabled=True):
        return self.nearest_pt_ind(pt,
                                   exclude_disabled_pts=exclude_disabled)[1]

    def nearest_pt_ind(self, pt, exclude_disabled_pts=True):
        if not exclude_disabled_pts:
            # This is just to warn that a more efficient implementation can be done in this case
            raise NotImplemented

        gpts = self.list_pts(exclude_disabled_pts=exclude_disabled_pts)
        ind = np.argmin(np.linalg.norm(gpts - pt, axis=1))
        return gpts[ind], ind

    def nearest_inds(self, pts, exclude_disabled=True):
        """Returns a list of the indices of the len(pts) gridpoints that lie
        closest to those in pts. 

        Args:            
            `pts`: num_pts x 3 array

        Returns:

            `l_inds`: list of `num_pts` distinct indices. It is sorted in 
            such a way that the first entry is the closest gridpt to any of 
            the pts, etc. 

        """

        # num_gridpts x num_pts
        dists = self.distances_to_grid_pts(pts,
                                           exclude_disabled=exclude_disabled)

        num_gridpts, num_pts = dists.shape

        l_inds = []
        # TODO: implement more efficiently
        for ind_pt in range(num_pts):
            row, col = np.unravel_index(np.argmin(dists), dists.shape)
            dists[row, :] = np.inf
            dists[:, col] = np.inf
            l_inds.append(row)

        return l_inds

    def distances_to_grid_pts(self, pts, exclude_disabled=True):
        """Returns a num_gridpts x num_pts matrix with the distances from each
        gridpt to each point in `pts`. 
        
        Args:
        
            `pts` is num_pts x 3."""

        pts = np.array(pts)

        # num_gridpts x 3
        t_gpts = self.list_pts(exclude_disabled_pts=exclude_disabled)
        # num_gridpts x 3 x 1
        t_gpts = t_gpts[:, :, None]

        # 3 x num_pts
        t_pts = pts.T
        # 1 x 3 x num_pts
        t_pts = t_pts[None, :, :]

        # num_gridpts x 3 x num_pts
        diffs = (t_gpts - t_pts)**2
        # num_gridpts x num_pts
        return np.sqrt(np.sum(diffs, axis=1))

    def t_coords_zplane(self):
        """Returns a 2 x num_pts_y x num_pts_x tensor whose first slice provides
        the x coordinates and the second slice provides the y coordinates of any
        horizontal plane, i.e., perpendicular to z."""
        return self.t_coords[0:2, 0, :, :]

    def slice_by_nearest_z(self, t_values, zval=None):
        """ `t_values` is a tensor whose last dimensions are num_pts_z x
        num_pts_y x num_pts_x. The returned tensor is a slice along the 3rd to
        last dimension associated with the closest z-coordinate to zval."""
        zind = self.nearest_z_ind(zval)
        return t_values[..., zind, :, :]

    def xfiber_by_nearest_yz(self, t_values, yval, zval):
        """ `t_values` is a tensor whose last dimensions are num_pts_z x
        num_pts_y x num_pts_x. The returned tensor is a fiber along the 1st
        dimension associated with the (y,z) coordinates that are closest to
        (yval,zval)."""
        yind = self.nearest_y_ind(yval)
        zind = self.nearest_z_ind(zval)
        return t_values[..., zind, yind, :]

    @property
    def x_axis(self):
        return np.arange(0, self.num_pts_x) * self.spacing[0]

    @property
    def y_axis(self):
        return np.arange(self.num_pts_y - 1, -1, step=-1) * self.spacing[1]

    @property
    def z_axis(self):
        return np.arange(0, self.num_pts_z) * self.spacing[2]

    @property
    def t_coords(self):
        """Returns:

        `t_c`: a `3 x num_pts_z x num_pts_y x num_pts_x` tensor where `t_c[0]`
        contains the x-coordinates of all grid points, `t_c[1]` the y
        coordinates, etc.

        """
        return self._axes_to_tensor(self.x_axis, self.y_axis, self.z_axis)

    @staticmethod
    def _axes_to_tensor(x, y, z):
        # x
        t_x_xyplane = np.repeat(x[None, None, :], len(y), axis=1)
        t_x = np.repeat(t_x_xyplane, len(z), axis=0)

        # y
        t_y_xyplane = np.repeat(y[None, :, None], len(x), axis=2)
        t_y = np.repeat(t_y_xyplane, len(z), axis=0)

        # z
        t_z_xzplane = np.repeat(z[:, None, None], len(x), axis=2)
        t_z = np.repeat(t_z_xzplane, len(y), axis=1)

        return np.array([t_x, t_y, t_z])

    @property
    def t_edges(self):
        """Returns:

        `t_e`: a `3 x (num_pts_z+1) x (num_pts_y+1) x (num_pts_x+1)` tensor
        containing the coordinates of the edges of the voxels. The voxels have
        width self.spacing[n] along the n-th dimension and are centered
        at the grid points. where `t_e[0]` contains the x-coordinates, `t_e[1]`
        the y coordinates, etc.

        """

        def centers_to_edges(vec, ind_coord):
            return np.concatenate((vec - self.spacing[ind_coord] / 2,
                                   [vec[-1] + self.spacing[ind_coord] / 2]))

        return self._axes_to_tensor(
            centers_to_edges(self.x_axis, 0),
            np.flip(centers_to_edges(np.flip(self.y_axis), 1)),
            centers_to_edges(self.z_axis, 2))

    @staticmethod
    def distance(pt_1, pt_2):
        assert pt_1.ndim == 1
        assert pt_2.ndim == 1
        return np.linalg.norm(pt_1 - pt_2)

    def list_pts(self, exclude_disabled_pts=True):
        """Returns an array that can be interpreted as a list of all the grid
        points. Specifically, if `b_exclude_disabled` is False, the array is
        `(num_pts_x * num_pts_y * num_pts_z) x 3`, where the first column
        contains the x-coordinate, etc. If `b_exclude_disabled`, the number of
        rows equals the number of enabled points.

        """

        return self.list_vals(self.t_coords, exclude_disabled_pts)

    def list_vals(self, t_vals, exclude_disabled_pts=True):
        """Args:

        `t_vals`: `N` x `num_pts_z` x `num_pts_y` x `num_pts_x` tensor where
        `t_vals[n, i, j, k]` is the n-th value that corresponds to the
        [i,j,k]-th point of the grid. This value will typically be a coordinate
        of the corresponding point or the value that a function takes at that
        point. `t_vals` can, therefore, encode a vector field of output
        dimension `N`

        Returns:

        `t_out` is a `self.num_pts` x `N` matrix if `b_exclude_disabled_pts==False`
        or if all points are enabled. The i-th row collects all values assigned
        to the i-th grid point. Else, `t_out` is a `num_enabled_pts` x `N`
        matrix. 

        """

        all_vals = self._grid_array_to_list(t_vals)

        if exclude_disabled_pts and self.t_enabled is not None:
            enable_vals = self._grid_array_to_list(self.t_enabled[None, ...])
            return all_vals[np.ravel(enable_vals)]

        return all_vals

    def _grid_array_to_list(self, t_vals):
        """Args:

        `t_vals`: `N` x `num_pts_z` x `num_pts_y` x `num_pts_x` tensor where
        `t_vals[n, i, j, k]` is the n-th value that corresponds to the
        [i,j,k]-th point of the grid. This value will typically be a coordinate
        of the corresponding point or the value that a function takes at that
        point. `t_vals` can, therefore, encode a vector field of output
        dimension `N`

        Returns:

        `t_out` is a `self.num_pts` x `N` matrix. The i-th row collects all
        values assigned to the i-th grid point. 
        """

        N = t_vals.shape[0]

        # t_vals is N x num_pts_z x num_pts_y x num_pts_x
        # => t_coords_1st_dim is num_pts_x x num_pts_z x num_pts_y x 3
        t_coords_1st_dim = np.transpose(t_vals, [3, 1, 2, 0])
        m_coords = np.reshape(t_coords_1st_dim, [self.num_pts, N])

        return m_coords

    def unlist_vals(self, m_vals):
        """Args:

        `m_vals` is a `self.num_enabled_pts` x `N` matrix. Typically, the i-th
        row corresponds to the value of a certain vector field at the
        i-th point of the grid.

        Returns:

        `t_out`: `N` x `num_pts_z` x `num_pts_y` x `num_pts_x` tensor
        where `t_out[n]` is the value of the n-th column of m_vals for
        the corresponding grid point. Disabled points will contain np.nan.

        """
        m_vals = np.array(m_vals)

        if self.t_enabled is not None:
            N = m_vals.shape[1]
            m_vals_all_gridpts = empty_array((self.num_pts, N))
            binds = self.list_vals(self.t_enabled[None, ...],
                                   exclude_disabled_pts=False)[:, 0]
            m_vals_all_gridpts[binds] = m_vals
        else:
            m_vals_all_gridpts = m_vals

        return self._list_to_grid_array(m_vals_all_gridpts)

    def _list_to_grid_array(self, m_vals):
        """Args:

        `m_vals` is a `self.num_pts` x `N` matrix. Typically, the i-th
        row corresponds to the value of a certain vector field at the
        i-th point of the grid.

        Returns:

        `t_out`: `N` x `num_pts_z` x `num_pts_y` x `num_pts_x` tensor
        where `t_out[n]` is the value of the n-th column of m_vals for
        the corresponding grid point. 

        """

        assert m_vals.ndim == 2
        assert m_vals.shape[0] == self.num_pts
        N = m_vals.shape[1]

        # m_vals is num_pts_z*num_pts_y*num_pts_x x N
        t_vals_1st_dim = np.reshape(
            m_vals, [self.num_pts_x, self.num_pts_z, self.num_pts_y, N])
        # => t_out is N x num_pts_z x num_pts_y x num_pts x
        t_out = np.transpose(t_vals_1st_dim, [3, 1, 2, 0])

        return t_out

    def random_pts(self, num_pts=1, z_val=None):
        """Returns a num_pts x 3 matrix. Each row is a random point drawn
        independently from a uniform distribution.

        if `z_val` is provided, then all generated points have
        z-coordinate `z_val`.

        """

        if z_val is None:
            raise NotImplementedError()

        x_coord = np.random.uniform(low=self.min_x,
                                    high=self.max_x,
                                    size=(num_pts, ))
        y_coord = np.random.uniform(low=self.min_y,
                                    high=self.max_y,
                                    size=(num_pts, ))
        z_coord = z_val * np.ones((num_pts, ))

        return np.array((x_coord, y_coord, z_coord)).T

    def field_line_integral(self, *args, mode="c", **kwargs):
        """This function is written in this class because of the dependence on
        the grid geometry. 

            Args:

            `t_values`: N x Nz x Ny x Nx matrix with the values of a field ot the 
            grid points. 

            `pt1` and `pt2` are points in the region. Their coordinates should not 
            stick outside from the region limits.

            `mode`: can be "c" or "python".

            Returns:

            vector with shape (N,), whose n-th entry is the integral of the n-th 
            scalar component of the vector field.

        """

        if mode == "c":
            return self.field_line_integral_c(*args, **kwargs)
        elif mode == "python":
            return self.field_line_integral_python(*args, **kwargs)
        else:
            raise ValueError()

    def field_line_integral_c(self, t_values, pt1, pt2):

        def check_or_raise(pt):
            # this function check whether or not a point is in the limits of the grid
            if not self.is_within_limits(pt):
                raise ValueError(
                    f"Point {pt} is not within the limits of the grid.")

        check_or_raise(pt1)
        check_or_raise(pt2)

        # t_values is currently a 4D tensor. The Python-C api cannot convert a matrix
        # which has more than 3 dimensions. So, we remove the first dimension of t_values,
        # which has only 1 element, hence t_values_reshaped is a 3D tensor.
        assert t_values.shape[
            0] == 1, "Line integration not implemented in C for vector fields. Use the Python version."
        #t_values_reshaped = np.squeeze(t_values)
        t_values_reshaped = t_values[0, ...]
        integral = grid_utilities.field_line_integral_C(
            self.spacing, t_values_reshaped, pt1, pt2, self.num_pts_y)

        return np.array([integral])

    def field_line_integral_python(self, t_values, pt1, pt2):

        def get_next_crossings():
            nc = (self.spacing *
                  (ind_current + 0.5 * increasing) - pt1) / pt_delta
            nc[increasing == 0] = 2  # constant entries will never be selected
            return nc

        def flip_y_ind(v_inds):
            v_inds[1] = self.num_pts_y - v_inds[1]
            return v_inds

        b_debug = False

        def check_or_raise(pt):
            if not self.is_within_limits(pt):
                raise ValueError(
                    f"Point {pt} is not within the limits of the grid.")

        check_or_raise(pt1)
        check_or_raise(pt2)

        increasing = ((pt2 - pt1) > 0).astype(int) - ((pt2 - pt1) < 0).astype(
            int)  # +1 if increasing, -1 if decreasing, 0 othw

        length = self.distance(pt1, pt2)

        pt_delta = pt2 - pt1
        pt_delta[increasing == 0] = 1  # to avoid dividing by 0

        ind_current = np.round(pt1 / self.spacing).astype(
            int)  # indices of current voxel. order: [x, Ny - y,z]

        t = 0
        integral = np.zeros((t_values.shape[0], ))
        if b_debug:
            all_crossings = [0]
            crossings_by_coord = [[], [], []]

        while t < 1:

            next_crossings = get_next_crossings()

            # take min across the 3 coords
            ind_coord_next_crossing = np.argmin(next_crossings)

            t_next = np.min((next_crossings[ind_coord_next_crossing], 1))

            if b_debug:
                pt_current = pt1 + (t_next + t) / 2 * (pt2 - pt1)
                inds_centroid_current = np.round(pt_current /
                                                 self.spacing).astype(int)
                if np.linalg.norm(inds_centroid_current - ind_current):
                    print("upssss")

            field_val = t_values[:, ind_current[2],
                                 self.num_pts_y - 1 - ind_current[1],
                                 ind_current[0]]
            if field_val != 0:  # Not sure if this check helps
                integral += (t_next - t) * length * field_val

            # update the current voxel
            t = t_next
            ind_current[ind_coord_next_crossing] += increasing[
                ind_coord_next_crossing]

            # debug
            if b_debug:
                all_crossings.append(t_next)
                if t < 1:
                    crossings_by_coord[ind_coord_next_crossing].append(t)

        if b_debug:

            def add_crossing_markers(t_vals, styles):
                v_t = np.array(t_vals)
                F.add_curve(xaxis=pt1[0] + v_t * (pt2[0] - pt1[0]) -
                            self.spacing[0] / 2,
                            yaxis=pt1[1] + v_t * (pt2[1] - pt1[1]) -
                            self.spacing[1] / 2,
                            styles=styles)

            F = GFigure()
            add_crossing_markers(all_crossings, "-")
            for crossings, styles in zip(crossings_by_coord, ["x", "v", "*"]):
                add_crossing_markers(crossings, styles)
            F.plot()
            plt.show()

        return integral

    def disable_by_indicator(self, f_indicator):
        # A grid point is disabled for a point with coords `coords` if
        # `f_indicator(coords)==True` or `f_indicator(coords) is not
        # empty/zero`. Else, the enabling status of the point is not altered.

        def disable_indicator(coords):
            val = f_indicator(coords)
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return val != 0
            val = np.array(val)
            return np.linalg.norm(val) != 0

        #disable_field = FunctionVectorField(grid=self, fun=disable_indicator)
        mask = self.unlist_vals([[disable_indicator(pt)]
                                 for pt in self.list_pts()])
        mask = np.logical_not(mask[0, ...])

        if self.t_enabled is None:
            self.t_enabled = mask
        else:
            self.t_enabled = np.logical_and(self.t_enabled, mask)

    def vertical_inds(self, increasing_z=True):
        """ Returns an iterator of lists. Each list contains the indices of the
        enabled grid points on a certain vertical, i.e., points with the same x and y.
        They are ordered in increasing or decreasing order of z depending on
        `increasing_z`."""

        for ind_x in range(self.num_pts_x):
            for ind_y in range(self.num_pts_y):
                if increasing_z:
                    yield [(ind_z, ind_y, ind_x)
                           for ind_z in range(self.num_pts_z)
                           if self.t_enabled[ind_z, ind_y, ind_x]]
                else:
                    yield [(ind_z, ind_y, ind_x)
                           for ind_z in range(self.num_pts_z - 1, -1, -1)
                           if self.t_enabled[ind_z, ind_y, ind_x]]

    def are_adjacent(self, pt_1, pt_2):
        """pt_1 and pt_2 are (3,) vectors """
        assert pt_1.ndim == 1
        assert pt_2.ndim == 1
        v_norm_dist = np.round(np.abs(pt_1 - pt_2) / self.spacing, 5)

        # For pt_1 to be adjacent to pt_2, all entries of v_norm_dist must be either 0 or 1.
        return set(v_norm_dist) <= {0, 1}

    def get_inds_adjacent(self, ref_pt, l_pts):
        """
        Returns a 1D array with the indices of the points in l_pts that are adjacent to `ref_pt`. 
        
        """
        assert ref_pt.ndim == 1
        m_pts = np.array(l_pts)
        assert m_pts.ndim == 2
        assert m_pts.shape[1] == ref_pt.shape[0]

        # If a row of m_comp contains only ones and zeros, the corresponding point is adjacent to m_conf_pt
        m_comp = np.round(
            np.abs(ref_pt[None, :] - m_pts) / self.spacing[None, :], 5)
        v_are_adjacent = np.all(m_comp * (m_comp - 1) == 0, axis=1)
        return np.where(v_are_adjacent)[0]
