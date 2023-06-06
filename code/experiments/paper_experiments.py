import numpy as np

from channels.tomographic_channel import TomographicChannel
from common.environment import BlockUrbanEnvironment
from common.utilities import watt_to_dbW

import gsim
from gsim.gfigure import GFigure

from relays.path_planners import PathPlanner, RandomRoadmapPathPlanner, SingleRelayMidpointPathPlanner, TwoRelaysAbovePathPlanner, UniformlySpreadRelaysPathPlanner

from simulators.path_planner_simulator import sim_min_ue_rate_metrics


class ExperimentSet(gsim.AbstractExperimentSet):

    def experiment_1034(l_args):
        """"
            This experiment plots
                + expected rate vs. time 

                + Tm(min_ue_rate) vs. min_ue_rate, where Tm(min_ue_rate) is the
                  MC average of the min time to reach a rate of min_ue_rate.
                  Those MC iterations where this min rate is never achieved are
                  not considered. 

                + Pfail(min_ue_rate) vs. min_ue_rate, where Pfail(min_ue_rate)
                  is the fraction of MC realizations where this min rate was
                  eventually achieved.
                
        """
        class CustomBlockUrbanEnvironment(BlockUrbanEnvironment):
            area_len = [100, 100, 70]
            block_limits_x = np.array([[10, 20], [40, 50], [70, 85]])
            block_limits_y = np.array([[10, 20], [40, 50], [70, 85]])

        env = CustomBlockUrbanEnvironment(num_pts_slf_grid=[50, 50, 20],
                                          num_pts_fly_grid=[12, 12, 8],
                                          min_fly_height=10,
                                          building_height=40,
                                          building_absorption=1)

        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=6e9,
                                     tx_dbpower=watt_to_dbW(.05),
                                     noise_dbpower=-97,
                                     bandwidth=20e6,
                                     min_link_capacity=2,
                                     max_link_capacity=7,
                                     antenna_dbgain_tx=12,
                                     antenna_dbgain_rx=12)

        fly_height = np.max(env.fly_grid.list_pts()[:, 2])
        min_uav_rate = 1e3
        samp_int = 1 / 20
        max_uav_speed = 2.5

        pp1 = SingleRelayMidpointPathPlanner(environment=env,
                                             channel=channel,
                                             min_uav_rate=min_uav_rate,
                                             fly_height=fly_height)

        pp2 = TwoRelaysAbovePathPlanner(environment=env,
                                        channel=channel,
                                        min_uav_rate=min_uav_rate,
                                        fly_height=fly_height)

        pp3 = UniformlySpreadRelaysPathPlanner(environment=env,
                                               channel=channel,
                                               num_uavs=2,
                                               min_uav_rate=min_uav_rate,
                                               fly_height=fly_height)

        ld_time_to_min_ue_rate = []
        ld_prob_of_failure = []
        ld_v_rate_vs_time = []

        v_min_ue_rate = np.array(
            [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 210]) * 1e6

        for min_ue_rate in v_min_ue_rate:

            print(f'min_ue_rate: {min_ue_rate}')

            pp4 = RandomRoadmapPathPlanner(environment=env,
                                           channel=channel,
                                           num_uavs=2,
                                           num_nodes=500,
                                           max_num_neighbors=50,
                                           min_uav_rate=min_uav_rate,
                                           fly_height=fly_height,
                                           mode_draw_conf_pt="feasible",
                                           mode_connect='los_n_rate',
                                           destination="min_ue_rate",
                                           min_ue_rate=min_ue_rate)

            l_planners = [pp1, pp2, pp3, pp4]

            d_time_to_min_ue_rate, d_prob_of_failure, d_v_ue_rates_vs_time = sim_min_ue_rate_metrics(
                env,
                channel,
                l_planners,
                samp_int=samp_int,
                max_uav_speed=max_uav_speed,
                padding=5,
                min_uav_rate=min_uav_rate,
                min_ue_rate=min_ue_rate,
                num_mc_iter=200)

            ld_time_to_min_ue_rate.append(d_time_to_min_ue_rate)
            ld_prob_of_failure.append(d_prob_of_failure)
            ld_v_rate_vs_time.append(d_v_ue_rates_vs_time)

        dl_time_to_min_ue_rate = ld_to_dl(ld_time_to_min_ue_rate)
        dl_prob_of_failure = ld_to_dl(ld_prob_of_failure)

        v_time_1, d_v_ue_rates_1 = get_d_v_ue_rate_vs_time(
            ld_v_rate_vs_time[5], samp_int)
        v_time_2, d_v_ue_rates_2 = get_d_v_ue_rate_vs_time(
            ld_v_rate_vs_time[7], samp_int)
        v_time_3, d_v_ue_rates_3 = get_d_v_ue_rate_vs_time(
            ld_v_rate_vs_time[9], samp_int)

        G1 = GFigure(ylabel="Mean Time [s]")

        G2 = GFigure(xlabel="Min. UE rate [Mbps]",
                     ylabel="Probability of Failure",
                     legend_loc="upper left")

        G3 = GFigure(xlabel="Time [s]",
                     ylabel="Mean UE Rate [Mbps]",
                     legend_loc="upper left")
        G4 = GFigure(xlabel="Time [s]",
                     ylabel="Mean UE Rate [Mbps]",
                     legend_loc="upper left")

        G5 = GFigure(xlabel="Time [s]",
                     ylabel="Mean UE Rate [Mbps]",
                     legend_loc="upper left")
        for key in dl_time_to_min_ue_rate.keys():
            G1.add_curve(xaxis=v_min_ue_rate / 1e6,
                         yaxis=dl_time_to_min_ue_rate[key])

            G2.add_curve(xaxis=v_min_ue_rate / 1e6,
                         yaxis=dl_prob_of_failure[key],
                         legend=key)

            G3.add_curve(xaxis=v_time_1,
                         yaxis=d_v_ue_rates_1[key] / 1e6,
                         legend=key)
            G4.add_curve(xaxis=v_time_2,
                         yaxis=d_v_ue_rates_2[key] / 1e6,
                         legend=key)
            G5.add_curve(xaxis=v_time_3,
                         yaxis=d_v_ue_rates_3[key] / 1e6,
                         legend=key)
        return [GFigure.concatenate([G1, G2], num_subplot_rows=2), G3, G4, G5]

    def experiment_1005(l_args):
        """"
            This experiment 
                + Uses UrbanEnvironment.random_pts_on_street to draw BS and UE
                  locations at random with specified distances. 
            
                + Average the time_to_connection across Monte Carlo realizations
                  for the algorithms vs. distance between the BS and UE.
        """
        class CustomBlockUrbanEnvironment(BlockUrbanEnvironment):
            area_len = [100, 100, 70]
            block_limits_x = np.array([[10, 20], [40, 50], [70, 85]])
            block_limits_y = np.array([[10, 20], [40, 50], [70, 85]])

        env = CustomBlockUrbanEnvironment(num_pts_slf_grid=[50, 50, 20],
                                          num_pts_fly_grid=[12, 12, 8],
                                          min_fly_height=10,
                                          building_height=40,
                                          building_absorption=1)
        
        channel = TomographicChannel(slf=env.slf,
                                     freq_carrier=6e9,
                                     tx_dbpower=watt_to_dbW(.05),
                                     noise_dbpower=-97,
                                     bandwidth=20e6,
                                     min_link_capacity=2,
                                     max_link_capacity=7,
                                     antenna_dbgain_tx=12,
                                     antenna_dbgain_rx=12)

        fly_height = np.max(env.fly_grid.list_pts()[:, 2])        
        min_uav_rate = 1e3
        samp_int = 1 / 20
        min_ue_rate = 160e6
        max_uav_speed = 2.5

        pp1 = SingleRelayMidpointPathPlanner(environment=env,
                                             channel=channel,
                                             min_uav_rate=min_uav_rate,
                                             fly_height=fly_height)

        pp2 = TwoRelaysAbovePathPlanner(environment=env,
                                        channel=channel,
                                        min_uav_rate=min_uav_rate,
                                        fly_height=fly_height)

        pp3 = UniformlySpreadRelaysPathPlanner(environment=env,
                                               channel=channel,
                                               num_uavs=2,
                                               min_uav_rate=min_uav_rate,
                                               fly_height=fly_height)

        pp4 = RandomRoadmapPathPlanner(environment=env,
                                       channel=channel,
                                       num_uavs=2,
                                       num_nodes=500,
                                       max_num_neighbors=50,
                                       min_uav_rate=min_uav_rate,
                                       fly_height=fly_height,
                                       mode_draw_conf_pt="feasible",
                                       mode_connect='los_n_rate',
                                       destination="min_ue_rate",
                                       min_ue_rate=min_ue_rate)

        l_planners = [pp1, pp2, pp3, pp4]

        ld_time_to_min_ue_rate = []
        ld_prob_of_failure = []
        ld_v_rate_vs_time = []

        v_bs_ue_dist = [20, 25, 30, 35, 40, 45, 50]

        for bs_ue_dist in v_bs_ue_dist:

            print(f'bs_ue_dist: {bs_ue_dist}')

            d_time_to_min_ue_rate, d_prob_of_failure, d_v_ue_rates_vs_time = sim_min_ue_rate_metrics(
                env,
                channel,
                l_planners=l_planners,
                samp_int=samp_int,
                max_uav_speed=max_uav_speed,
                padding=5,
                min_uav_rate=min_uav_rate,
                min_ue_rate=min_ue_rate,
                num_mc_iter=200,
                bs_ue_dist=bs_ue_dist)

            ld_time_to_min_ue_rate.append(d_time_to_min_ue_rate)
            ld_prob_of_failure.append(d_prob_of_failure)
            ld_v_rate_vs_time.append(d_v_ue_rates_vs_time)

        dl_time_to_min_ue_rate = ld_to_dl(ld_time_to_min_ue_rate)
        dl_prob_of_failure = ld_to_dl(ld_prob_of_failure)

        v_time, d_v_ue_rates = get_d_v_ue_rate_vs_time(ld_v_rate_vs_time[1],
                                                       samp_int)

        G1 = GFigure(ylabel="Mean Time [s]")

        G2 = GFigure(xlabel="Distance BS-UE [m]",
                     ylabel="Probability of Failure",
                     legend_loc="upper left")

        G3 = GFigure(xlabel="Time [s]",
                     ylabel="Mean UE Rate [Mbps]",
                     legend_loc="upper left")
        for key in dl_time_to_min_ue_rate.keys():
            G1.add_curve(xaxis=v_bs_ue_dist, yaxis=dl_time_to_min_ue_rate[key])

            G2.add_curve(xaxis=v_bs_ue_dist,
                         yaxis=dl_prob_of_failure[key],
                         legend=key)

            G3.add_curve(xaxis=v_time,
                         yaxis=d_v_ue_rates[key] / 1e6,
                         legend=key)
        return [GFigure.concatenate([G1, G2], num_subplot_rows=2), G3]

def get_d_v_ue_rate_vs_time(d_v_ue_rates, samp_int):
    """Pads the time series so that all have the same length"""
    num_time_samples = 0
    for key in d_v_ue_rates:
        if all(np.isnan(d_v_ue_rates[key])):
            continue
        num_time_samples = np.maximum(num_time_samples, len(d_v_ue_rates[key]))

    v_time = np.linspace(0, num_time_samples, num_time_samples) * samp_int

    # Padding the last element
    for key in d_v_ue_rates:
        if num_time_samples == 0:
            d_v_ue_rates[key] = np.array([])
        else:
            d_v_ue_rates[key] = np.concatenate(
                (d_v_ue_rates[key],
                 np.array([d_v_ue_rates[key][-1]] *
                          (num_time_samples - len(d_v_ue_rates[key])))),
                axis=0)

    return v_time, d_v_ue_rates


def from_graph(m_nodes, m_adjacency):
    """ 
    Args:

    `m_nodes`: num_nodes x 2

    `m_adjacency`: num_nodes x num_nodes

    """

    G = GFigure(xaxis=m_nodes[:, 0], yaxis=m_nodes[:, 1], styles='.')
    for ind_row in range(m_adjacency.shape[0]):
        for ind_col in range(m_adjacency.shape[0]):
            if m_adjacency[ind_row, ind_col]:
                # Join points m_nodes[ind_row] and m_nodes[ind_col]
                G.add_curve(xaxis=[m_nodes[ind_row, 0], m_nodes[ind_col, 0]],
                            yaxis=[m_nodes[ind_row, 1], m_nodes[ind_col, 1]],
                            styles=['c'])

    return G


def plot_path(G, m_nodes, path_to_source, color='k'):

    G.add_curve(xaxis=m_nodes[:, 0], yaxis=m_nodes[:, 1], styles='.')

    for ind in range(len(path_to_source) - 1):

        pt1 = path_to_source[ind]
        pt2 = path_to_source[ind + 1]
        G.add_curve(xaxis=[m_nodes[pt1, 0], m_nodes[pt2, 0]],
                    yaxis=[m_nodes[pt1, 1], m_nodes[pt2, 1]],
                    styles=[color])

    return G


def plot_all_rates_vs_time(channel, min_uav_rate, lm_path, bs_loc, ue_loc):
    """Plots (better, returns a GFigure) the user rate (see email) vs. time. """

    # Compute the rates from lm_path
    m_uavs_rate, v_ue_rate = PathPlanner.rate_from_path(
        channel, lm_path, min_uav_rate, bs_loc, ue_loc)

    v_x = np.linspace(1, len(lm_path), len(lm_path))
    v_min_uav_rate = min_uav_rate * np.ones((m_uavs_rate[0].shape[0], )) / 1e6

    l_legend = [f"UAV {ind_uav}" for ind_uav in range(m_uavs_rate.shape[0])
                ] + ["UE", "Min uav rate"]

    G = GFigure(
        xaxis=v_x,
        yaxis=[m_uavs_rate[ind] / 1e6 for ind in range(m_uavs_rate.shape[0])] +
        [v_ue_rate / 1e6] + [v_min_uav_rate],
        xlabel="Time step",
        ylabel="Rate [Mbps]",
        title="Rate vs time step",
        legend=l_legend)

    return G


def ld_to_dl(ld):
    dl = dict()
    for d in ld:
        for key in d.keys():
            if key not in dl.keys():
                dl[key] = []
            dl[key].append(d[key])
    return dl