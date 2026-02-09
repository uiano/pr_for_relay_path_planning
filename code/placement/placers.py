from common.grid import RectangularGrid3D

import logging

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
