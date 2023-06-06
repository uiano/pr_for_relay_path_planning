import abc
from IPython.core.debugger import set_trace
import numpy as np
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from gsim.gfigure import GFigure

from common.fields import FunctionVectorField
from common.utilities import natural_to_dB, dbm_to_watt, watt_to_dbm
from channels.channel import Channel

import logging

log = logging.getLogger("channel")
log.setLevel(logging.DEBUG)

speed_of_light = 3e8

class TomographicChannel(Channel):
    integral_mode = 'c' # Can be 'c' or 'python'
    def __init__(self, *args, slf=None, nesh_scaling=False, **kwargs):

        assert slf is not None

        super().__init__(*args, **kwargs)

        assert slf.output_length == 1
        self.slf = slf
        self.nesh_scaling = nesh_scaling

    def dbabsorption(self, pt_1, pt_2):
        """Returns the attenuation due to absorption between points pt_1 and
        pt_2. """

        integral = self.slf.line_integral(pt_1, pt_2, mode=self.integral_mode)[0]

        if self.nesh_scaling:
            integral = integral/np.sqrt(np.linalg.norm(pt_2-pt_1))

        return integral

    # def weight(self, pt_1, pt_2):
    #     """Returns a VectorField indicating the weight that should be given to
    #     the SLF at each grid point when computing the absorption between pt_1
    #     and pt_2. """

    #     def f_dists(pt):
    #         return [np.linalg.norm(pt - pt_1), np.linalg.norm(pt - pt_2)]
    #     fl_dist = FunctionVectorField(grid=self.grid, fun=f_dists)

    #     def f_w_of_dists(m_dists):
    #         phi1 = self.grid.distance(pt_1, pt_2)
    #         phi2 = np.sum(m_dists, axis=1) # sum of

    def dbgain(self, pt_1, pt_2):
        """ See parent."""

        friis = super().dbgain_free_space(pt_1, pt_2)
        absorption = self.dbabsorption(pt_1, pt_2)

        return friis - absorption
