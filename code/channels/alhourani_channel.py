import numpy as np
from channels.channel import Channel

import logging

log = logging.getLogger("channel")
log.setLevel(logging.DEBUG)

speed_of_light = 3e8


class AlHouraniChannel(Channel):
    """ Al-Hourani channel model [alhourani2014lap]
        
    Args:
        dbatten_factor_los= \eta_los in [alhourani2014lap]

        dbatten_factor_nlos=\eta_nlos in [alhourani2014lap]

        env_const_a=a in [alhourani2014lap]

        env_const_b=b in [alhourani2014lap]

        Use [boryaliniz2016] to see possible values of const_a, const_b, \eta_los, and \eta_nlos.
             

    """

    def __init__(self,
                 *args,
                 dbatten_factor_los=1,
                 dbatten_factor_nlos=20,
                 env_const_a=None,
                 env_const_b=None,
                 env=None,
                 **kwargs):
        """
        Args:

                env: object of class UrbanEnvironment. It should be None if
                env_const_a or env_const_b are provided. If not None, then the
                constants a and b are taken from `env`.        
        """

        super().__init__(*args, **kwargs)

        assert (env is not None) ^ (env_const_a is not None
                                    and env_const_b is not None)

        self.dbatten_factor_los = dbatten_factor_los
        self.dbatten_factor_nlos = dbatten_factor_nlos

        if env is None:
            self.env_const_a = env_const_a
            self.env_const_b = env_const_b
        else:
            self.env_const_a, self.env_const_b = self.get_a_b_params(env)

    def dbloss_los_n_nlos(self, pt_1, pt_2):
        """
            Returns the mean additional loss between two points, one of them is
            a UAV and the other a ground user. 

            """
        vert_dist = np.abs(pt_1[2] - pt_2[2])

        horiz_dist = np.sqrt((pt_1[0] - pt_2[0])**2 + (pt_1[1] - pt_2[1])**2)
        elevation_angle = np.degrees(np.arctan(vert_dist / horiz_dist))

        prob_los = 1 / (1 + self.env_const_a *
                        np.exp(-self.env_const_b *
                               (elevation_angle - self.env_const_a)))

        prob_nlos = 1 - prob_los

        dbloss_los = self.dbatten_factor_los * prob_los
        dbloss_nlos = self.dbatten_factor_nlos * prob_nlos

        return dbloss_los + dbloss_nlos

    def dbgain(self, pt_1, pt_2):
        """ See parent."""

        friis = super().dbgain_free_space(pt_1, pt_2)

        return friis - self.dbloss_los_n_nlos(pt_1, pt_2)

    @staticmethod
    def get_a_b_params(env):
        """
            Args:

                env: object of class UrbanEnvironment.

        """

        def get_alpha_beta_gamma(env):
            """
            Args:

                env: object of class UrbanEnvironment.

            Returns:

                param_alpha: the ratio of built-up land area to the total land area
                (dimensionless).
                
                param_beta: the mean number of buildings per unit area
                (buildings/km2).

                param_gamma: an estimate of the mean of the Rayleigh disribution of
                the building heights (meters).
            
            """

            area_building_total = env.building_area
            area_land = env.area_len[0] * env.area_len[1]
            param_alpha = area_building_total / area_land

            # beta: the mean number of buildings per unit area (buildings/km2).
            param_beta = len(env.buildings) / (area_land / 1e6)

            # gamma: estimated here using the method of moments
            param_gamma = env.mean_building_height / np.sqrt(np.pi / 2)

            return param_alpha, param_beta, param_gamma

        def get_fitting_param(param_alpha, param_beta, param_gamma,
                              m_poly_coeffs):

            param_mul = param_alpha * param_beta

            param = 0

            for ind_row in range(m_poly_coeffs.shape[0]):
                for ind_col in range(m_poly_coeffs.shape[1] - ind_row):
                    param += m_poly_coeffs[ind_row, ind_col] * (
                        param_mul**ind_col) * (param_gamma**ind_row)

            return param

        # compute param_a and param_b from param_alpha, param_beta, and param_gamma
        m_poly_coeffs_a = np.array([[9.34e-1, 2.30e-1, -2.25e-3, 1.86e-5],
                                    [1.97e-2, 2.44e-3, 6.58e-6, 0],
                                    [-1.24e-4, -3.34e-6, 0, 0],
                                    [2.73e-7, 0, 0, 0]])
        m_poly_coeffs_b = np.array([[1.17e0, -7.56e-2, 1.98e-3, -1.78e-5],
                                    [-5.79e-3, 1.81e-4, -1.65e-6, 0],
                                    [1.73e-5, -2.02e-7, 0, 0],
                                    [-2.00e-8, 0, 0, 0]])

        param_alpha, param_beta, param_gamma = get_alpha_beta_gamma(env)

        param_a = get_fitting_param(param_alpha, param_beta, param_gamma,
                                    m_poly_coeffs_a)
        param_b = get_fitting_param(param_alpha, param_beta, param_gamma,
                                    m_poly_coeffs_b)

        return param_a, param_b
