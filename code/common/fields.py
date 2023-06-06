import abc
from IPython.core.debugger import set_trace
import numpy as np
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from gsim.gfigure import GFigure
from common.utilities import is_vector
"""The classes in this file should not be dependent on the grid geometry. If
   your algorithm depends on the geometry of the grid, code it in grid.py.  """


class Field():

    def __init__(self, grid=None, t_values=None, name="Field"):

        self.grid = grid
        self.name = name
        self.t_values = t_values

    def clone(self):

        return type(self)(grid=self.grid.clone(),
                          t_values=np.copy(self.t_values),
                          name=self.name)


class VectorField(Field):

    def map_on_grid(self, fun):
        """Args:
        
        - `fun` is a function taking points as input and returning an
           array with shape (N,). If the array contains Boolean values, they are
           turned into numeric (this is because np.nan is used and a Boolean
           array cannot contain np.nan entries.)

        Returns:

        - `t_out`: tensor of shape `N x num_pts_z x num_pts_y x num_pts_x` where
          `t_out[n]` contains the values of the first entry returned by `fun` at
          all grid points. 

        """

        all_pts = self.grid.list_pts()

        # TODO: use np.vectorize or map.
        all_vals = np.array([fun(pt) for pt in all_pts])

        return self.grid.unlist_vals(all_vals)

    def line_integral(self, pt1, pt2, **kwargs):
        """Returns an approximation of the line integral of the function between
        pt1 and pt2."""
        return self.grid.field_line_integral(self.t_values, pt1, pt2, **kwargs)

    def plot_z_slices(self, zvals=None, znum=None):
        """ 
            Args: One and only one of the following arguments must be specified.

            `zvals` list or 1D array of values for z

            `znum` number of values of z. In this case, `znum` slices are taken
            uniformly between min_z and max_z
        """

        if znum is not None:
            assert zvals is None
            zvals = np.linspace(self.grid.min_z, self.grid.max_z, znum)
        else:
            assert zvals is not None

        xy_coords = self.grid.t_coords_zplane()

        z_slices = np.array([
            self.grid.slice_by_nearest_z(self.t_values, zval=zval)
            for zval in zvals
        ])

        all_vals = np.ravel(z_slices)
        all_vals = all_vals[np.logical_not(np.isnan(all_vals))]
        max_val = all_vals.max()
        min_val = all_vals.min()

        G = GFigure(num_subplot_columns=2,
                    global_color_bar=True,
                    global_color_bar_label=self.name)
        for zval, z_slice in zip(zvals, z_slices):

            G.next_subplot(title=f"z = {self.grid.nearest_z(zval):.2f}",
                           xlabel="x [m]",
                           ylabel=f"y [m]",
                           grid=False,
                           zlim=(min_val, max_val))
            G.add_curve(xaxis=xy_coords[0, :, :],
                        yaxis=xy_coords[1, :, :],
                        zaxis=z_slice[0],
                        zinterpolation="none")
            # G.add_curve(xaxis=m_measurement_loc[:, 0],
            #             yaxis=m_measurement_loc[:, 1],
            #             styles="+w")

        return G

    def plot_as_blocks(self):
        """The voxels where the field is not zero are filled. This is done for
        each scalar field component.

        See https://matplotlib.org/3.1.0/gallery/mplot3d/voxels.html if you wish
        to extend this code to select the color of each component.

        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        t_edges = self.grid.t_edges

        for t_values_slice in self.t_values:
            t_indicator = (t_values_slice > 0)
            ax.voxels(t_edges[0],
                      t_edges[1],
                      t_edges[2],
                      t_indicator,
                      edgecolor='k')
        plt.xlabel("x")
        plt.ylabel("y")

        return ax

    @property
    def output_length(self):
        assert self.t_values.ndim == 4
        return self.t_values.shape[0]

    @staticmethod
    def sum_fields(l_fields):
        tt_values = np.array([field.t_values for field in l_fields])
        t_values = np.sum(tt_values, axis=0)
        return VectorField(grid=l_fields[0].grid,
                           t_values=t_values,
                           name="Sum")

    @staticmethod
    def min_over_fields(l_fields):
        """Returns another field whose value at each point is the minimum of the
        values at that point taken by each of the fields in `l_fields`."""

        tt_values = np.array([field.t_values for field in l_fields])
        t_values = np.min(tt_values, axis=0)
        return VectorField(grid=l_fields[0].grid,
                           t_values=t_values,
                           name="Min")

    def arg_coord_max(self):
        """ Returns the coordinates of the point with largest value of the field"""

        assert self.output_length == 1
        pts = self.grid.list_pts(exclude_disabled_pts=True)
        vals = self.grid.list_vals(self.t_values, exclude_disabled_pts=True)

        return pts[np.argmax(vals)]

    @staticmethod
    def concatenate(l_fields):
        """Returns a VectorField whose output is the result of concatenating the
        outputs of the fields in `l_fields`."""

        t_values = np.concatenate([field.t_values for field in l_fields],
                                  axis=0)
        return VectorField(grid=l_fields[0].grid,
                           t_values=t_values,
                           name="Concatenation")

    def list_vals(self, exclude_disabled_pts=True):
        """ 
            Returns:

            `t_out` is a `self.num_pts` x `N` matrix if `b_exclude_disabled_pts==False`
            or if all points are enabled. The i-th row collects all values assigned
            to the i-th grid point. Else, `t_out` is a `self.grid.num_enabled_pts` x `N`
            matrix. """
        return self.grid.list_vals(self.t_values,
                                   exclude_disabled_pts=exclude_disabled_pts)

    def disable_gridpts_by_dominated_verticals(self):
        """"Disables each grid point (x,y,z) if there is another grid point
        (x,y,z') with z'<z satisfying that F(x,y,z')>= F(x,y,z) entrywise. """
        for vertical in self.grid.vertical_inds(increasing_z=False):
            for ind_current in range(len(vertical) - 1):
                current = vertical[ind_current]
                for ind_next in range(ind_current + 1, len(vertical)):
                    next = vertical[ind_next]

                    if all(self.t_values[:, next[0], next[1], next[2]] >= self.
                           t_values[:, current[0], current[1], current[2]]):
                        self.grid.t_enabled[current[0], current[1],
                                            current[2]] = False
                        break

    def clip(self, upper=None):
        self.t_values = np.minimum(self.t_values, upper)
        return self


class FunctionVectorField(VectorField):

    def __init__(self, *args, fun=None, **kwargs):

        super().__init__(*args, **kwargs)

        assert self.grid, "A grid must be provided"

        assert fun is not None
        assert is_vector(
            fun(self.grid.origin
                )), "Argument `fun` must be a function that returns a vector"

        self.t_values = self.map_on_grid(fun)
