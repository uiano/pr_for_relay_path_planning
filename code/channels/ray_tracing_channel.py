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


class RayTracingChannel(Channel):

    def __init__(self, *args, env=None, **kwargs):

        super().__init__(*args, **kwargs)

        if env is not None:
            self.env = env
        else:
            raise ValueError

    def dbgain(self, pt_user, pt_fly_grid):
        """Returns the gain in dB between positions `pt_user` (user grid point)
        and `pt_fly_grid` (fly grid point). """

        # ind_fly_grid_pt = self.env.coords_to_ind_fly_grid_pt(pt_2)
        # ind_user_pt = self.env.coords_to_ind_user_pt(pt_1)

        # return self.env.m_ch_gains[ind_fly_grid_pt, ind_user_pt]

        return self.env.ch_dbgain(pt_fly_grid, pt_user)
