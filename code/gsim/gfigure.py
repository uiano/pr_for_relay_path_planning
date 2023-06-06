import sys
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import copy
import numpy as np

title_to_caption = False
default_figsize = None  # `None` lets plt choose
"""

The easiest way to learn how to use this module is to run the examples at the
end of this file. To do so, cd to the folder gsim and type:

python3 gfigure.py <figure_number>

where <figure_number> is an integer. See the code and possible values of
<figure_number> in `plot_example_figure` below.

"""
""" 
TODO: 

Replace lists of a numeric type in xaxis or yaxis with numpy
arrays. With lists it gets messy when using 3D plots. 

"""


def hist_bin_edges_to_xy(hist, bin_edges):
    """ PDF estimate from a histogram with bins of possibly different lengths. """

    def duplicate_entries(v_in):
        """ If v_in = [v1,v2,...vN], this function returns [v1, v1, v2, v2, ..., vN, vN]."""
        return np.ravel(np.tile(v_in, (2, 1)).T)

    v_bin_widths = bin_edges[1:] - bin_edges[:-1]
    v_p = hist / np.sum(hist) / v_bin_widths

    v_x = duplicate_entries(bin_edges)
    v_y = np.concatenate(([0], duplicate_entries(v_p), [0]))
    return v_x, v_y


def is_number(num):
    #return isinstance(num, (int, float, complex, bool))
    # From https://stackoverflow.com/questions/500328/identifying-numeric-and-array-types-in-numpy

    if hasattr(num, "numpy"):
        num = num.numpy()

    if isinstance(num, np.ndarray):
        if num.size != 1:
            return False

    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(num, attr) for attr in attrs)


class Curve:

    def __init__(self,
                 xaxis=None,
                 yaxis=[],
                 zaxis=None,
                 zinterpolation='none',
                 ylower=[],
                 yupper=[],
                 style=None,
                 mode=None,
                 legend_str=""):
        """

      See GFigure.__init__ for more information.

      1. For 2D plots:
      ---------------

      xaxis : None or a list of a numeric type. In the latter case, its length 
          equals the length of yaxis.

      yaxis : list of a numeric type. 

      zaxis : None

      ylower, yupper: [] or lists of a numeric type with the same length as yaxis.

      mode : can be 'plot' or 'stem'

      2. For 3D plots:
      ----------------

      xaxis: M x N numpy array

      yaxis: M x N numpy array

      zaxis: M x N numpy array

      zinterpolation: see GFigure.__init__

      Other arguments
      ---------------

      style : str used as argument to plt.plot()

      """

        # Input check
        if zaxis is None:
            # 2D plot
            if type(yaxis) != list:
                set_trace()
                raise TypeError("`yaxis` must be a list of numeric entries")
            if type(xaxis) == list:
                assert len(xaxis) == len(yaxis)
            elif xaxis is not None:
                raise TypeError(
                    "`xaxis` must be a list of numeric entries or None")
        else:
            # 3D plot

            # zaxis
            if not isinstance(zaxis, np.ndarray):
                raise TypeError(f"Argument `zaxis` must be of class np.array.")
            if zaxis.ndim != 2:
                raise ValueError(f"Argument `zaxis` must be of dimension 2. ")

            num_rows, num_cols = zaxis.shape

            # xaxis and yaxis
            def is_empty(arg):
                return (arg is None) or ((type(arg) == list) and
                                         (len(arg) == 0))

            if is_empty(xaxis):
                assert is_empty(
                    yaxis), "If `xaxis` is empty, then `yaxis` must be empty"
            else:
                if isinstance(xaxis, np.ndarray):
                    assert isinstance(
                        yaxis, np.ndarray
                    ), "If `xaxis` is an `np.ndarray`, then `yaxis` must be an `np.ndarray`."

                    # At this point, both are arrays. Just check their dimensions
                    if xaxis.ndim == 1:
                        assert yaxis.ndim == 1, "If `xaxis.ndim` is 1, then `yaxis.ndim` must also be 1."
                        assert xaxis.shape == (
                            num_cols,
                        ), f"If `xaxis.ndim` is 1, then `xaxis.shape` must be ({num_cols},). "
                        assert yaxis.shape == (
                            num_rows,
                        ), f"If `yaxis.ndim` is 1, then `yaxis.shape` must be ({num_rows},). "
                    elif xaxis.ndim == 2:
                        assert yaxis.ndim == 2, "If `xaxis.ndim` is 2, then `yaxis.ndim` must also be 2."
                        assert xaxis.shape == (
                            num_rows, num_cols
                        ), f"If `xaxis.ndim` is 2, then `xaxis.shape` must be ({num_rows},{num_cols}). "
                        assert yaxis.shape == (
                            num_rows, num_cols
                        ), f"If `yaxis.ndim` is 2, then `yaxis.shape` must be ({num_rows},{num_cols}). "
                    else:
                        raise ValueError("`xaxis.ndim` must be either 1 or 2.")
                else:
                    raise TypeError(
                        f"If `xaxis` is not empty, it must be of class `np.ndarray`."
                    )

        if (style is not None) and (type(style) != str):
            raise TypeError("`style` must be of type str or None")
        if type(legend_str) != str:
            raise TypeError("`legend_str` must be of type str")

        # Common
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.mode = mode

        # 2D
        self.ylower = ylower
        self.yupper = yupper
        self.style = style
        self.legend_str = legend_str

        # 3D
        self.zaxis = zaxis
        self.zinterpolation = zinterpolation
        self.image = None

    def __repr__(self):
        return f"<Curve: legend_str = {self.legend_str}, num_points = {len(self.yaxis)}>"

    def plot(self, **kwargs):

        if self.is_3D:
            self._plot_3D(**kwargs)
        else:
            self._plot_2D()

    def _plot_2D(self):

        def plot_band(lower, upper):
            if self.xaxis:
                plt.fill_between(self.xaxis, lower, upper, alpha=0.2)
            else:
                plt.fill_between(lower, upper, alpha=0.2)

        if hasattr(self, "ylower"):  # check for backwards compatibility
            if self.ylower:
                plot_band(self.ylower, self.yaxis)
            if self.yupper:
                plot_band(self.yaxis, self.yupper)

        # if type(self.xaxis) == list and len(self.xaxis):
        #     if self.style:
        #         plt.plot(self.xaxis,
        #                  self.yaxis,
        #                  self.style,
        #                  label=self.legend_str)
        #     else:
        #         plt.plot(self.xaxis, self.yaxis, label=self.legend_str, use_line_collection=True)
        # else:
        #     if self.style:
        #         plt.plot(self.yaxis, self.style, label=self.legend_str)
        #     else:
        #         plt.plot(self.yaxis, label=self.legend_str)
        if type(self.xaxis) == list and len(self.xaxis):
            axis_args = (self.xaxis, self.yaxis)
        else:
            axis_args = (self.yaxis, )

        if hasattr(self, 'mode') and (self.mode is not None) and (self.mode
                                                                  == 'stem'):

            def plot_fun(*args, **kwargs):
                return plt.stem(*args, **kwargs, use_line_collection=True)
        else:
            plot_fun = plt.plot

        style = self.style if self.style else "-"

        plot_fun(*axis_args, style, label=self.legend_str)

    def _plot_3D(self, axis=None, interpolation="none", zlim=None):

        assert axis

        # Default mode
        if not hasattr(self, 'mode') or (self.mode is None):
            self.mode = 'imshow'

        len_y, len_x = self.zaxis.shape

        # xaxis and yaxis
        if not isinstance(self.xaxis, np.ndarray):
            v_x = np.arange(len_x)
            v_y = np.arange(len_y)
            m_X, m_Y = np.meshgrid(v_x, v_y)
            m_Z = self.zaxis
        else:
            if (self.xaxis.ndim == 1):
                m_X, m_Y = np.meshgrid(self.xaxis, self.yaxis)
                m_Z = self.zaxis
            else:
                m_X, m_Y = self.xaxis, self.yaxis
                m_Z = self.zaxis

        if self.mode == 'imshow':
            self.image = axis.imshow(
                m_Z,
                interpolation=self.zinterpolation,
                cmap='jet',
                # origin='lower',
                extent=[m_X[-1, 0], m_X[-1, -1], m_Y[-1, 0], m_Y[0, 0]],
                vmax=zlim[1] if zlim else None,
                vmin=zlim[0] if zlim else None)
        elif self.mode == 'contour3D':
            axis.contour3D(m_X, m_Y, m_Z, 50, cmap='plasma')
        elif self.mode == 'surface':
            axis.plot_surface(m_X,
                              m_Y,
                              m_Z,
                              rstride=1,
                              cstride=1,
                              cmap='viridis',
                              edgecolor='none')

        else:
            raise ValueError(f'Unrecognized 3D plotting mode. Got {self.mode}')

    def legend_is_empty(l_curves):

        for curve in l_curves:
            if curve.legend_str != "":
                return False
        return True

    @property
    def projection(self):
        """This is used to create the axes.
        
        Note that plt is not consistent. The projection mode can be '3d'
        (lowercase), but the function is called 'contour3D'. 
        """

        if self.is_3D:
            if hasattr(self, 'mode') and (self.mode == 'contour3D'
                                          or self.mode == 'surface'):
                return '3d'
        return None

    @property
    def is_3D(self):
        return hasattr(self, "zaxis") and self.zaxis is not None


class Subplot:

    def __init__(self,
                 title="",
                 xlabel="",
                 ylabel="",
                 zlabel="",
                 color_bar=False,
                 grid=True,
                 xlim=None,
                 ylim=None,
                 zlim=None,
                 legend_loc=None,
                 create_curves=True,
                 **kwargs):
        """
      For a description of the arguments, see GFigure.__init__

      """

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.color_bar = color_bar
        self.grid = grid
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.legend_loc = legend_loc

        self.l_curves = []
        if create_curves:
            self.add_curve(**kwargs)

    def __repr__(self):
        return f"<Subplot objet with title=\"{self.title}\", len(self.l_curves)={len(self.l_curves)} curves>"

    def is_empty(self):

        return not any([self.title, self.xlabel, self.ylabel, self.l_curves])

    def update_properties(self, **kwargs):

        if "title" in kwargs:
            self.title = kwargs["title"]
        if "xlabel" in kwargs:
            self.xlabel = kwargs["xlabel"]
        if "ylabel" in kwargs:
            self.ylabel = kwargs["ylabel"]
        if "zlabel" in kwargs:
            self.ylabel = kwargs["zlabel"]

    def add_curve(self,
                  xaxis=[],
                  yaxis=[],
                  zaxis=None,
                  zinterpolation="bilinear",
                  ylower=[],
                  yupper=[],
                  styles=[],
                  mode=None,
                  legend=tuple()):
        """
      Adds one or multiple curves to `self`. See documentation of GFigure.__init__
      """

        if zaxis is None:
            # 2D figure
            self.l_curves += Subplot._l_2D_curves_from_input_args(xaxis,
                                                                  yaxis,
                                                                  ylower,
                                                                  yupper,
                                                                  styles,
                                                                  legend,
                                                                  mode=mode)
        else:
            # 3D figure
            self.l_curves.append(
                Curve(xaxis=xaxis,
                      yaxis=yaxis,
                      zaxis=zaxis,
                      zinterpolation=zinterpolation,
                      mode=mode))

    def _l_2D_curves_from_input_args(xaxis, yaxis, ylower, yupper, styles,
                                     legend, mode):

        # Process the subplot input.  Each entry of xaxis can be
        # either None (use default x-axis) or a list of float. Each
        # entry of yaxis is a list of float. Both xaxis and
        # yaxis will have the same length.
        l_xaxis, l_yaxis = Subplot._list_from_axis_arguments(xaxis, yaxis)
        # Each entry of `l_ylower` and `l_yupper` is either None (do
        # not shade any area) or a list of float.
        l_ylower, _ = Subplot._list_from_axis_arguments(ylower, yaxis)
        l_yupper, _ = Subplot._list_from_axis_arguments(yupper, yaxis)
        l_style = Subplot._list_from_style_argument(styles)
        # Note: all these lists can be empty.

        # Process style input.
        if len(l_style) == 0:
            l_style = [None] * len(l_xaxis)
        elif len(l_style) == 1:
            l_style = l_style * len(l_xaxis)
        else:
            #   if len(l_style) < len(l_xaxis):
            #       raise ValueError("The length of the styles argument needs to be at least the number of curves.")
            assert len(l_style) >= len(l_xaxis), "The length of `style` must be"\
                  " either 1 or no less than the number of curves"
            l_style = l_style[0:len(l_xaxis)]

        # Process the legend
        assert ((type(legend) == tuple) or (type(legend) == list)
                or (type(legend) == str))
        if type(legend) == str:
            legend = [legend] * len(l_xaxis)
        else:  # legend is tuple or list
            if len(legend) == 0:
                legend = [""] * len(l_xaxis)
            else:
                if type(legend[0]) != str:
                    raise TypeError(
                        "`legend` must be an str, list of str, or tuple of str."
                    )
                if (len(legend) != len(l_yaxis)):
                    raise ValueError(
                        f"len(legend)={len(legend)} should equal 0 or the "
                        f"number of curves={len(l_yaxis)}")

        b_debug = True
        if b_debug:
            conditions = [
                len(l_xaxis) == len(l_yaxis),
                len(l_xaxis) == len(l_style),
                type(l_xaxis) == list,
                type(l_yaxis) == list,
                type(l_style) == list,
                (len(l_xaxis) == 0) or (type(l_xaxis[0]) == list)
                or (l_xaxis[0] is None),
                (len(l_yaxis) == 0) or (type(l_yaxis[0]) == list)
                or (l_yaxis[0] is None),
                (len(l_style) == 0) or (type(l_style[0]) == str)
                or (l_style[0] is None),
            ]
            if not np.all(conditions):
                print(conditions)
                set_trace()

        # Construct Curve objects
        l_curve = []
        for xax, yax, ylow, yup, stl, leg in zip(l_xaxis, l_yaxis, l_ylower,
                                                 l_yupper, l_style, legend):
            l_curve.append(
                Curve(xaxis=xax,
                      yaxis=yax,
                      ylower=ylow,
                      yupper=yup,
                      style=stl,
                      legend_str=leg,
                      mode=mode))
        return l_curve

    def _list_from_style_argument(style_arg):
        """
      Returns a list of str. 
      """
        err_msg = "Style argument must be an str "\
            "or list of str"
        if type(style_arg) == str:
            return [style_arg]
        elif type(style_arg) == list:
            for entry in style_arg:
                if type(entry) != str:
                    raise TypeError(err_msg)
            return copy.copy(style_arg)
        else:
            raise TypeError(err_msg)

    def _list_from_axis_arguments(xaxis_arg, yaxis_arg):
        """Processes subplot arguments and returns two lists of the same length
      whose elements can be either None or lists of a numerical
      type. None means "use the default x-axis for this curve".

      Both returned lists can be empty if no curve is specified.

      """

        def unify_format(axis):

            def ndarray_to_list(arr):
                """Returns a list of lists."""
                assert (type(arr) == np.ndarray)
                if arr.ndim == 1:
                    if len(arr):
                        return [list(arr)]
                    else:
                        return []
                elif arr.ndim == 2:
                    return [[arr[row, col] for col in range(0, arr.shape[1])]
                            for row in range(0, arr.shape[0])]
                else:
                    raise ValueError(
                        "Input arrays need to be of dimension 1 or 2")

            # Compatibility with TensorFlow
            if hasattr(axis, "numpy"):
                axis = axis.numpy()

            if (type(axis) == np.ndarray):
                return ndarray_to_list(axis)
            elif (type(axis) == list):
                # at this point, `axis` can be:
                # 1. empty list: either no curves are specified or, in case of
                #    the x-axis, the specified curves should use the default xaxis.
                if len(axis) == 0:
                    return []
                # 2. A list of a numeric type. Only one axis specified.
                if is_number(axis[0]):
                    #return [copy.copy(axis)]
                    return [[float(ax) for ax in axis]]
                # 3. A list where each entry specifies one axis.
                else:
                    out_list = []
                    for entry in axis:
                        # Each entry can be:
                        # 3a. a tf.Tensor
                        if hasattr(entry, "numpy"):
                            entry = entry.numpy()

                        # 3b. an np.ndarray
                        if isinstance(entry, np.ndarray):
                            if entry.ndim == 1:
                                #out_list.append(copy.copy(entry))
                                out_list.append([float(ent) for ent in entry])
                            else:
                                raise Exception(
                                    "Arrays inside the list must be 1D in the current implementation"
                                )
                        # 3c. a list of a numeric type
                        elif type(entry) == list:
                            # 3c1: for an x-axis, empty `entry` means default axis.
                            if len(entry) == 0:
                                out_list.append([])
                            # 3c2: Numerical type
                            elif is_number(entry[0]):
                                #out_list.append(copy.copy(entry))
                                out_list.append([float(ent) for ent in entry])
                            else:
                                raise TypeError
                    return out_list
            elif axis is None:
                return [None]
            else:
                raise TypeError

        # Construct two lists of possibly different lengths.
        l_xaxis = unify_format(xaxis_arg)
        l_yaxis = unify_format(yaxis_arg)
        """At this point, `l_xaxis` can be:
      - []: use the default xaxis if a curve is provided (len(l_yaxis)>0). 
        No curves specified if len(l_yaxis)=0. 
      - [None]: use the default xaxis for all specfied curves.
      - [xaxis1, xaxis2,... xaxisN], where xaxisn is a list of float.
      """

        # Expand (broadcast) l_xaxis to have the same length as l_yaxis
        str_message = "Number of curves in the xaxis must be"\
            " 1 or equal to the number of curves in the y axis"
        if len(l_xaxis) > 1 and len(l_yaxis) != len(l_xaxis):
            raise Exception(str_message)
        if len(l_xaxis) == 0 and len(l_yaxis) > 0:
            l_xaxis = [None]
        if len(l_yaxis) > 1:
            if len(l_xaxis) == 1:
                l_xaxis = l_xaxis * len(l_yaxis)
            if len(l_xaxis) != len(l_yaxis):
                raise Exception(str_message)
        elif len(l_yaxis) == 1:
            if len(l_xaxis) != 1:
                raise Exception(str_message)

        return l_xaxis, l_yaxis

    def plot(self, **kwargs):

        for curve in self.l_curves:
            curve.plot(
                zlim=self.zlim
                if hasattr(self, "zlim") else None,  # backwards comp.
                **kwargs)

        if not Curve.legend_is_empty(self.l_curves):
            if not hasattr(self, "legend_loc"):
                self.legend_loc = None  # backwards compatibility
            plt.legend(loc=self.legend_loc)

        # Axis labels
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        if self.projection == '3d' and hasattr(self, 'zlabel') and self.zlabel:
            plt.gca().set_zlabel(self.zlabel)

        # Color bar
        if hasattr(self, "color_bar") and self.color_bar:
            image = self.get_image()
            if image is None:
                raise ValueError(
                    "color_bar=True but no color figure was specified")
            cbar = plt.colorbar(image)  #, cax=cbar_ax)
            if self.zlabel:
                cbar.set_label(self.zlabel)

        if self.title:
            plt.title(self.title)

        if "grid" in dir(self):  # backwards compatibility
            plt.grid(self.grid)

        if "xlim" in dir(self):  # backwards compatibility
            if self.xlim:
                plt.xlim(self.xlim)

        if "ylim" in dir(self):  # backwards compatibility
            if self.ylim:
                plt.ylim(self.ylim)

        return

    def get_image(self):
        """Scans l_curves to see if one has defined the attribute "image". If
      so, it returns the value of this attribute, else it returns
      None.

      """
        for curve in self.l_curves:
            if curve.image:
                return curve.image
        return None

    @property
    def projection(self):
        """This is used to create the axes."""
        for curve in self.l_curves:
            if curve.projection == '3d':
                return '3d'
        return None

    @property
    def is_3D(self):
        """Returns true if at least one curve is 3D."""
        for curve in self.l_curves:
            if curve.is_3D:
                return True
        return False


class GFigure:
    str_caption = None

    def __init__(self,
                 *args,
                 figsize=None,
                 ind_active_subplot=0,
                 num_subplot_rows=None,
                 num_subplot_columns=1,
                 global_color_bar=False,
                 global_color_bar_label="",
                 global_color_bar_position=[0.85, 0.35, 0.02, 0.5],
                 layout="tight",
                 **kwargs):
        """

      FIGURE
      ======

       figsize: can be a tuple of format (width, height), e.g. (20., 10.). If
            None and the global `default_figsize` is not None, the value of the
            latter is used.

      `layout`: can be "", "tight", or "constrained". See pyplot documentation.
       
                Since April 2022, layout='tight' is set by default.

        One of `num_subplot_rows` or `num_subplot_columns` can be specified for
        figures with multiple subplots. 

      SUBPLOT ARGUMENTS:
      =================

      The first set of arguments allow the user to create a subplot when
      creating the GFigure object.

      title : str 

      xlabel : str

      ylabel : str

      grid : bool

      xlim : tuple, endpoints for the x axis.

      ylim : tuple, endpoints for the y axis.

      zlim : tuple, endpoints for the z axis. Used e.g. for the color scale. 

      legend_loc: str, it indicates the location of the legend. Example values:
          "lower left", "upper right", etc.

      CURVE ARGUMENTS:
      =================

      1. 2D plots
         -----------

      xaxis and yaxis: 

        (a) To specify only one curve: 

            - `yaxis` can be a 1D np.ndarray, a 1D tf.Tensor or a list of a
              numeric
            type 

            - `xaxis` can be None, a list of a numeric type, or a 1D np.array
            of the same length as `yaxis`. 

        (b) To specify one or more curves: 

            - `yaxis` can be: -> a list whose elements are as described in (a)
              -> M
            x N np.ndarray or tf.Tensor. Each row corresponds to a curve. 

            - `xaxis` can be either as in (a), so all curves share the same
              X-axis
            points, or -> a list whose elements are as described in (a) -> Mx x
            N np.ndarray. Each row corresponds to a curve. Mx must be either M
            or 1.
          
      ylower and yupper: specify a shaded area around the curve, used e.g.
        for confidence bounds. The area between ylower and yaxis as well as the
        area between yaxis and yupper are shaded. Their format is the same as
        yaxis.

      zaxis: None

      mode: it can be 'plot' (default) or 'stem'

      2. 3D plots
      -----------

        2a. Axes
        --------
        
        zaxis: M x N numpy array. When `mode` is 'imshow', the bottom left of
        the matrix corresponds to the bottom left of the figure. 

        There are 3 options:
      
        - xaxis and yaxis are M x N numpy arrays. The (x,y) coordinates
          corresponding to zaxis[i,j] are xaxis[i,j] and yaxis[i,j].

        - xaxis and yaxis are vectors of length N and M, respectively. The (x,y)
          coordinates corresponding to zaxis[i,j] are xaxis[j] and yaxis[i].
          This is useful e.g. when we want the matrix to provide the values of a
          function on the first quadrant, where the bottom-left entry of the
          matrix would correspond to the origin and yaxis is thought of as a
          column vector whose bottom entry provides the y-coordinate of the
          origin. 

        - xaxis and yaxis are None or []. In this case, it is understood that
          the user wants to visualize the entries of a matrix. Thus, the (x,y)
          coordinates corresponding to zaxis[i,j] are respectively j and i.
          Arguments xlabel and ylabel respectively correspond to columns and
          rows. 
      

        2b. Rest of arguments
        ---------------------

      mode: it can be 'imshow' (default), 'contour3D', or 'surface'.

      zinterpolation: Supported values are 'none', 'antialiased', 'nearest',
      'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
      'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel',
      'mitchell', 'sinc', 'lanczos'.

      color_bar: If True, a color bar is created for the specified axis.

      global_color_bar: if True, one color bar for the entire figure. 

      global_color_bar_label: str indicating the label of the global color bar.

      global_color_bar_position: vector with four entries.

      3. Others
         ---------

      styles: specifies the style argument to plot, as in MATLAB. Possibilities:
          - str : this style is applied to all curves specified by `xaxis` and
          `yaxis` will - list of str : then style[n] is applied to the n-th
          curve. Its length must be at least the number of curves.

      legend : str, tuple of str, or list of str. If the str begins with "_",
          then that curve is not included in the legend.


      ARGUMENTS FOR SPECIFYING HOW TO SUBPLOT:
      ========================================


     `ind_active_subplot`: The index of the subplot that is created and where
          new curves will be added until a different value for the property of
          GFigure with the same name is specified. A value of 0 refers to the
          first subplot.

      `num_subplot_rows` and `num_subplot_columns` determine the number of
          subplots in each column and row respectively. If None, their value is
          determined by the value of the other of these parameters and the
          number of specified subplots. If the number of specified subplots does
          not equal num_subplot_columns*num_subplot_rows, then the value of
          num_subplot_columns is determined from the number of subplots and
          num_subplot_rows.

          The values of the properties of GFigure with the same name can be
          specified subsequently.


      """

        # Create a subplot if the arguments specify one
        new_subplot = Subplot(*args, **kwargs)
        self.ind_active_subplot = ind_active_subplot
        if not new_subplot.is_empty():
            # List of axes to create subplots
            self.l_subplots = [None] * (self.ind_active_subplot + 1)
            self.l_subplots[self.ind_active_subplot] = new_subplot
        else:
            self.l_subplots = []

        self.num_subplot_rows = num_subplot_rows
        self.num_subplot_columns = num_subplot_columns
        self.figsize = figsize
        self.global_color_bar = global_color_bar
        self.global_color_bar_label = global_color_bar_label
        self.global_color_bar_position = global_color_bar_position

        if layout == "" or layout == "tight":
            self.layout = layout
        else:
            raise ValueError("Invalid value of argument `layout`")

    def __repr__(self):
        return f"<GFigure object with len(self.l_subplots)={len(self.l_subplots)} subplots>"

    def add_curve(self, *args, ind_active_subplot=None, **kwargs):
        """
         Similar arguments to __init__ above.


      """

        # Modify ind_active_subplot only if provided
        if ind_active_subplot is not None:
            self.ind_active_subplot = ind_active_subplot

        self.select_subplot(self.ind_active_subplot, **kwargs)
        self.l_subplots[self.ind_active_subplot].add_curve(*args, **kwargs)

    def next_subplot(self, **kwargs):
        # Creates a new subplot at the end of the list of axes. One can
        # specify subplot parameters; see GFigure.
        self.ind_active_subplot = len(self.l_subplots)
        if kwargs:
            self.l_subplots.append(Subplot(**kwargs))

    def select_subplot(self, ind_subplot, **kwargs):
        """Creates the `ind_subplot`-th subplot if it does not exist and
        selects it. Subplot keyword parameters can also be provided (see
        GFigure), but parameters to create a curve are ignored.        
        """

        self.ind_active_subplot = ind_subplot

        # Complete the list l_subplots if index self.ind_active_subplot does
        # not exist.
        if ind_subplot >= len(self.l_subplots):
            self.l_subplots += [None] * (self.ind_active_subplot -
                                         len(self.l_subplots) + 1)

        # Create if it does not exist
        if self.l_subplots[self.ind_active_subplot] is None:
            self.l_subplots[self.ind_active_subplot] = Subplot(
                create_curves=False, **kwargs)
        else:
            self.l_subplots[self.ind_active_subplot].update_properties(
                **kwargs)

    def plot(self):

        # backwards compatibility
        if "figsize" not in dir(self):
            figsize = None
        else:
            figsize = self.figsize
        if figsize is None:
            figsize = default_figsize

        F = plt.figure(figsize=figsize)

        # Determine the number of rows and columns for arranging the subplots
        num_axes = len(self.l_subplots)
        if self.num_subplot_rows is not None:
            self.num_subplot_columns = int(
                np.ceil(num_axes / self.num_subplot_rows))
        else:  # self.num_subplot_rows is None
            if self.num_subplot_columns is None:
                # Both are None. Just arrange thhe plots as a column
                self.num_subplot_columns = 1
                self.num_subplot_rows = num_axes
            else:
                self.num_subplot_rows = int(
                    np.ceil(num_axes / self.num_subplot_columns))

        # Process title
        if title_to_caption and (len(self.l_subplots) == 1):
            self.str_caption = self.l_subplots[0].title
            self.l_subplots[0].title = ""
            print("Caption: ", self.str_caption)

        # Actual plotting operation
        for index, subplot in enumerate(self.l_subplots):
            axis = plt.subplot(self.num_subplot_rows,
                               self.num_subplot_columns,
                               index + 1,
                               projection=self.l_subplots[index].projection)
            if self.l_subplots[index] is not None:

                self.l_subplots[index].plot(axis=axis)

        # Layout
        if hasattr(self, "layout"):  # backwards compatibility
            if self.layout == "":
                pass
            elif self.layout == "tight":
                plt.tight_layout()
            else:
                raise ValueError("Invalid value of argument `layout`")

        # Color bar
        if hasattr(self, "global_color_bar") and self.global_color_bar:

            for subplot in self.l_subplots:
                image = subplot.get_image()
                if image:
                    break
            F.subplots_adjust(right=0.85)

            cbar_ax = F.add_axes(self.global_color_bar_position)
            cbar = F.colorbar(image, cax=cbar_ax)

            if self.global_color_bar_label:
                cbar.set_label(self.global_color_bar_label)

        return F

    def concatenate(it_gfigs, num_subplot_rows=None, num_subplot_columns=1):
        """Concatenates the subplots of a collection of GFigure objects.

     Args:
       it_gfigs: iterable that returns GFigures. 

       num_subplot_rows and num_subplot_columns: see GFigure.__init__()

     Returns: 
       gfig: an object of class GFigure.

     """

        l_subplots = [
            subplot for gfig in it_gfigs for subplot in gfig.l_subplots
        ]

        gfig = next(iter(it_gfigs))  # take the first
        gfig.l_subplots = l_subplots
        gfig.num_subplot_rows = num_subplot_rows
        gfig.num_subplot_columns = num_subplot_columns

        return gfig

    def export(self, base_filename):
        # Save figure to pdf
        filename_pdf = base_filename + ".pdf"
        print(f"Exporting GFigure as {filename_pdf}")
        plt.savefig(filename_pdf)

        # Save caption if applicable
        if hasattr(self, "str_caption") and self.str_caption is not None:
            basename_txt = base_filename + ".txt"
            print(f"Saving caption as {basename_txt}")
            with open(basename_txt, "w") as f:
                f.write(self.str_caption)

    @staticmethod
    def show():
        plt.show()


def plot_example_figure(ind_example):
    """
    This is example code to learn how to use GFigure. See the description at the
    top of this file. 
    
    """

    v_x = np.linspace(0, 10, 20)
    v_y1 = v_x**2 - v_x + 3
    v_y2 = v_x**2 + v_x + 3
    v_y3 = v_x**2 - 2 * v_x - 10

    if ind_example == 1:
        # Example with a single curve, single subplot
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabolas",
                    legend="P1")

    elif ind_example == 2:
        # Example with three curves on one subplot
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabolas",
                    legend="P1")
        G.add_curve(xaxis=v_x, yaxis=v_y2, legend="P2")
        G.add_curve(xaxis=v_x, yaxis=v_y3, legend="P3")

    elif ind_example == 3:
        # Typical scheme where a simulation function produces each
        # curve.
        def my_simulation():
            coef = np.random.random()
            v_y_new = coef * v_y1
            G.add_curve(xaxis=v_x, yaxis=v_y_new, legend="coef = %.2f" % coef)

        """ One can specify the axis labels and title when the figure is
        created."""
        G = GFigure(xlabel="x", ylabel="f(x)", title="Parabola")
        for ind in range(0, 6):
            my_simulation()

    elif ind_example == 4:
        # Example with two subplots
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabolas",
                    legend="P1")
        G.add_curve(xaxis=v_x, yaxis=v_y2, legend="P2")
        G.next_subplot(xlabel="x")
        G.add_curve(
            xaxis=v_x,
            yaxis=v_y3,
            legend="P3",
        )

    elif ind_example == 5:
        # Example with a large multiplot
        G = GFigure(num_subplot_rows=4)
        for ind in range(0, 12):
            G.select_subplot(ind, xlabel="x", ylabel="f(x)", title="Parabolas")
            G.add_curve(xaxis=v_x, yaxis=v_y1, legend="P1", styles="r")

    elif ind_example == 6:
        # Typical scheme where a simulation function produces each subplot
        def my_simulation():
            G.next_subplot(xlabel="x", ylabel="f(x)", title="Parabola")
            G.add_curve(xaxis=v_x, yaxis=v_y1, legend="P1", styles="r")

        """ Important not to specify axis labels or the title in the next line
        because that would create an initial subplot without curves
        and, therefore, function `next_subplot` will move to the
        second subplot of the figure the first time `my_simulation` is
        executed."""

        G = GFigure(num_subplot_rows=3)
        for ind in range(0, 6):
            my_simulation()

    elif ind_example == 7:
        # Colorplot of a function of 2 arguments.
        num_points_x = 30
        num_points_y = 30
        gridpoint_spacing = 1 / 30
        v_x_coords = np.arange(0, num_points_x) * gridpoint_spacing
        v_y_coords = np.arange(num_points_y - 1, -1,
                               step=-1) * gridpoint_spacing
        x_coords, y_coords = np.meshgrid(v_x_coords, v_y_coords, indexing='xy')

        def my_simulation():
            xroot = np.random.random()
            yroot = np.random.random()
            zaxis = (x_coords - xroot)**2 + (y_coords - yroot)**2
            G.next_subplot(xlabel="x",
                           ylabel="y",
                           zlabel="z",
                           grid=False,
                           color_bar=False,
                           zlim=(0, 1))
            G.add_curve(xaxis=x_coords, yaxis=y_coords, zaxis=zaxis)
            G.add_curve(xaxis=[xroot], yaxis=[yroot], styles="+w")

        G = GFigure(num_subplot_rows=3,
                    global_color_bar=True,
                    global_color_bar_label="z")

        for ind in range(0, 6):
            my_simulation()

    elif ind_example == 8:
        # Scatter plot
        v_x = np.linspace(0, 10, 200)
        v_y = v_x + np.random.normal(size=(len(v_x), ))
        G = GFigure(xaxis=v_x,
                    yaxis=v_y,
                    styles=['.'],
                    xlabel='x',
                    ylabel='y=x+w')

    elif ind_example == 9:
        # 3D plots
        G = GFigure(num_subplot_rows=3)

        # The three modes 'imshow' (default), 'contour3d', and 'surface' are
        # tested.

        # 1. Auto axes
        # Format as an image
        m_Z = np.reshape(np.arange(20 * 30), (20, 30))
        G.next_subplot(
            zaxis=m_Z,
            xlabel='columns',
            ylabel='rows',
        )
        G.next_subplot(zaxis=m_Z,
                       xlabel='columns',
                       ylabel='rows',
                       zlabel='value',
                       mode='contour3D')

        # 2. Semi-manual specification of the axes (most typical)
        xaxis = np.linspace(0, 20, 50)
        yaxis = np.flip(np.linspace(0, 10, 50))
        zaxis = yaxis[:, None]**2 + 10 * xaxis[None, :]
        G.next_subplot(
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=zaxis,
            xlabel='x',
            ylabel='y',
        )
        G.next_subplot(xaxis=xaxis,
                       yaxis=yaxis,
                       zaxis=zaxis,
                       xlabel='x',
                       ylabel='y',
                       zlabel='z',
                       mode='surface')

        # 3. Fully manual specification of the axes
        v_xaxis = np.linspace(0, 6, 50)
        v_yaxis = np.linspace(0, np.pi, 50)
        m_X, m_Y = np.meshgrid(v_xaxis, v_yaxis)
        m_Z = np.sin(m_Y) * m_X
        G.next_subplot(
            xaxis=m_X,
            yaxis=m_Y,
            zaxis=m_Z,
            xlabel='x',
            ylabel='y',
        )
        G.next_subplot(xaxis=m_X,
                       yaxis=m_Y,
                       zaxis=m_Z,
                       xlabel='x',
                       ylabel='y',
                       zlabel='z',
                       mode='contour3D')

    G.plot()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("""Usage from command line: 
$ python3 gfigure.py <example_index>
            
where <example_index> is an integer. See function `example_figures`.""")
    else:
        plot_example_figure(int(sys.argv[1]))
