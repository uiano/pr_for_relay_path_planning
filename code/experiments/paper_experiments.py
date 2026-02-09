import numpy as np
from channels.tomographic_channel import TomographicChannel
from common.environment import BlockUrbanEnvironment

from common.utilities import watt_to_dbW

import gsim
from gsim.gfigure import GFigure

from relays.path_planners import RandomRoadmapPathPlanner, SingleRelayMidpointPathPlanner, UniformlySpreadRelaysPathPlanner, SegmentationPathPlanner, SingleRelayAlhouraniMaxRatePathPlanner, TwoRelaysOneMidOneAboveUEPathPlanner, TwoRelaysMaxRatePathPlanner

from simulators.path_planner_simulator import sim_metrics_vs_min_ue_rate_2serve_ue, sim_optimality_vs_dist_bs_static_ue

import os


class ExperimentSet(gsim.AbstractExperimentSet):
    """
        67. Experiments to compare algorithms planning a path to serve a static
            UE
    """

    # vs dist_bs_ue: time to connect, frac. outage time, total transferred data
    def experiment_6725(l_args):
        """
        This experiment plots
            + Time to connect: Tm(min_ue_rate) vs. dist_bs_ue, where
              Tm(min_ue_rate) is the MC average of the min time to reach a rate
              of min_ue_rate. Those MC iterations where this min rate is never
              achieved are not considered.
            
            + Frac. outage time: the MC average of the ratios of i) the number
              of time instants in which the user does not have min_ue_rate to
              ii) the total number of time instants.
            
            + Total transferred data: expected rate vs. time.
        
        """

        exp_name = "exp6725"
        save_at = './output/relay_placement_experiments/' + exp_name + '/'
        os.makedirs(save_at, exist_ok=True)

        b_run_new = True
        ind_mc_iter_start = 0
        num_mc_iter = 1

        min_uav_rate = 200e3

        max_uav_speed = 7  # m/s
        samp_int = 2
        ue_speed = 2
        num_pts_ue_grid = [50, 50, 1]
        time_duration = samp_int  # static user
        bs_loc = np.array([20, 470, 0.])

        prpp_max_num_neighbors = 100
        prpp_num_nodes = 2000
        prpp_destination = "min_ue_rate"

        ldv_rate_vs_time = []
        ld_time_to_min_ue_rate = []
        ld_prob_failure = []
        ldv_cumsum_data_over_time = []
        ld_average_rate = []

        min_ue_rate = 90e6
        # [70, 100, 150, 200, 250, 300, 350, 400]
        v_dist_bs_ue = np.array([70, 100, 150, 200, 250, 300, 350, 400])

        f_env = lambda: CustomBlockUrbanEnvironment(
            num_pts_slf_grid=[50, 50, 20],
            num_pts_fly_grid=[12, 12, 8],  # [12, 12, 8]
            min_fly_height=10,
            building_height=40,
            building_absorption=1)
        f_channel = lambda env: TomographicChannel(slf=env.slf,
                                                   freq_carrier=6e9,
                                                   tx_dbpower=watt_to_dbW(.05),
                                                   noise_dbpower=-97,
                                                   bandwidth=20e6,
                                                   min_link_capacity=2,
                                                   max_link_capacity=7,
                                                   antenna_dbgain_tx=12,
                                                   antenna_dbgain_rx=12)

        fly_height = f_env().fly_grid.list_pts()[:, 2].max()

        lf_planners = [
            lambda env, channel: SingleRelayMidpointPathPlanner(
                environment=env,
                channel=channel,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Zeng et al.'),
            #
            lambda env, channel: SingleRelayAlhouraniMaxRatePathPlanner(
                environment=env,
                channel=channel,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Ghazzai et al.'),
            #
            lambda env, channel: TwoRelaysOneMidOneAboveUEPathPlanner(
                environment=env,
                channel=channel,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Lee et al.'),
            #
            lambda env, channel: TwoRelaysMaxRatePathPlanner(
                environment=env,
                channel=channel,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Zhang et al.'),
            #
            lambda env, channel: UniformlySpreadRelaysPathPlanner(
                environment=env,
                channel=channel,
                num_uavs=2,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Yanmaz et al.'),
            #
            lambda env, channel: RandomRoadmapPathPlanner(
                environment=env,
                channel=channel,
                num_uavs=2,
                num_nodes=prpp_num_nodes,
                max_num_neighbors=prpp_max_num_neighbors,
                min_uav_rate=min_uav_rate,
                mode_draw_conf_pt="feasible",
                mode_connect='min_rate_only',
                destination=prpp_destination,
                min_ue_rate=min_ue_rate,
                b_conf_pts_meet_min_ue_rate=False,
                b_tentative=True,
                name_custom='PRFI (Tentative)'),
            #
            lambda env, channel: RandomRoadmapPathPlanner(
                environment=env,
                channel=channel,
                num_uavs=2,
                num_nodes=prpp_num_nodes,
                max_num_neighbors=prpp_max_num_neighbors,
                min_uav_rate=min_uav_rate,
                mode_draw_conf_pt="feasible",
                mode_connect='min_rate_only',
                destination=prpp_destination,
                min_ue_rate=min_ue_rate,
                b_conf_pts_meet_min_ue_rate=False,
                b_tentative=False,
                name_custom='PRFI')
        ]

        for dist_bs_ue in v_dist_bs_ue:
            print('-' * 20)
            print(f'Distance BS - UE: {dist_bs_ue} m')
            save_single_pt_at = save_at + f'distBSUE{dist_bs_ue:003}' + '/'
            os.makedirs(save_single_pt_at, exist_ok=True)

            d_time_to_min_ue_rate, d_prob_failure, d_prob_outage, dv_ue_rates_vs_time, dv_cumsum_data_over_time, d_average_rate = sim_metrics_vs_min_ue_rate_2serve_ue(
                f_env,
                f_channel,
                lf_planners,
                samp_int=samp_int,
                ue_speed=ue_speed,
                max_uav_speed=max_uav_speed,
                num_pts_ue_grid=num_pts_ue_grid,
                time_duration=time_duration,
                min_uav_rate=min_uav_rate,
                min_ue_rate=min_ue_rate,
                ind_mc_iter_start=ind_mc_iter_start,
                num_mc_iter=num_mc_iter,
                loc_bs=bs_loc,
                dist_bs_ue=dist_bs_ue,
                save_at=save_single_pt_at,
                b_run_new=b_run_new,
                b_return_failure_only=True)

            ldv_rate_vs_time.append(dv_ue_rates_vs_time)
            ld_time_to_min_ue_rate.append(d_time_to_min_ue_rate)
            ld_prob_failure.append(d_prob_failure)
            ldv_cumsum_data_over_time.append(dv_cumsum_data_over_time)
            ld_average_rate.append(d_average_rate)

        dl_time_to_min_ue_rate = ld_to_dl(ld_time_to_min_ue_rate)
        dl_prob_outage = None
        dl_prob_failure = ld_to_dl(ld_prob_failure)
        dl_average_rate = ld_to_dl(ld_average_rate)

        return plot_performance(dl_time_to_min_ue_rate,
                                dl_prob_outage,
                                dl_prob_failure,
                                dl_average_rate,
                                ldv_rate_vs_time,
                                ldv_cumsum_data_over_time,
                                samp_int=samp_int,
                                xlabel="Distance BS-UE [m]",
                                xticks=v_dist_bs_ue,
                                l_min_ue_rate=min_ue_rate)

    # vs min_ue_rate: time to connect, frac. outage time, total transferred data
    def experiment_6739(l_args):
        """
        This experiment plots
            + Time to connect: Tm(min_ue_rate) vs. min_ue_rate, where
              Tm(min_ue_rate) is the MC average of the min time to reach a rate
              of min_ue_rate. Those MC iterations where this min rate is never
              achieved are not considered.
            
            + Prob. of failure: the MC average of the ratios of i) the number of
              time instants in which the user does not have min_ue_rate to ii)
              the total number of time instants.
            
            + Mean rate vs. time.
        """

        exp_name = "exp6739"
        save_at = './output/relay_placement_experiments/' + exp_name + '/'

        b_run_new = True
        ind_mc_iter_start = 0
        num_mc_iter = 1

        min_uav_rate = 200e3

        max_uav_speed = 7  # m/s
        samp_int = 2
        ue_speed = 2
        num_pts_ue_grid = [50, 50, 1]
        time_duration = samp_int  # static user
        bs_loc = np.array([20, 470, 0.])

        # parameters for probabilistic roadmap path planners
        prpp_max_num_neighbors = 100
        prpp_num_nodes = 2000
        prpp_destination = "min_ue_rate"

        ldv_rate_vs_time = []
        ld_time_to_min_ue_rate = []
        ld_prob_failure = []
        ldv_cumsum_data_over_time = []
        ld_average_rate = []

        # [10, 30, 50, 70, 90]
        v_min_ue_rate = np.array([10, 30, 50, 70, 90]) * 1e6

        f_env = lambda: CustomBlockUrbanEnvironment(
            num_pts_slf_grid=[50, 50, 20],
            num_pts_fly_grid=[12, 12, 8],  # 12, 12, 8
            min_fly_height=10,
            building_height=40,  # 
            building_absorption=1)
        f_channel = lambda env: TomographicChannel(slf=env.slf,
                                                   freq_carrier=6e9,
                                                   tx_dbpower=watt_to_dbW(.05),
                                                   noise_dbpower=-97,
                                                   bandwidth=20e6,
                                                   min_link_capacity=2,
                                                   max_link_capacity=7,
                                                   antenna_dbgain_tx=12,
                                                   antenna_dbgain_rx=12)

        fly_height = f_env().fly_grid.list_pts()[:, 2].max()

        for min_ue_rate in v_min_ue_rate:
            print('-' * 20)
            print(f'\nMin_ue_rate: {min_ue_rate}')
            save_single_pt_at = save_at + f'minUeRate{int(min_ue_rate/1e6):003}' + '/'
            os.makedirs(save_single_pt_at, exist_ok=True)

            lf_planners = [
                lambda env, channel: SingleRelayMidpointPathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Zeng et al.'),
                #
                lambda env, channel: SingleRelayAlhouraniMaxRatePathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Ghazzai et al.'),
                #
                lambda env, channel: TwoRelaysOneMidOneAboveUEPathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Lee et al.'),
                #
                lambda env, channel: TwoRelaysMaxRatePathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Zhang et al.'),
                #
                lambda env, channel: UniformlySpreadRelaysPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Yanmaz et al.'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=prpp_num_nodes,
                    max_num_neighbors=prpp_max_num_neighbors,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    b_tentative=True,
                    name_custom='PRFI (Tentative)'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=prpp_num_nodes,
                    max_num_neighbors=prpp_max_num_neighbors,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    b_tentative=False,
                    name_custom='PRFI'),
                #
            ]

            d_time_to_min_ue_rate, d_prob_failure, d_prob_outage, dv_ue_rates_vs_time, dv_cumsum_data_over_time, d_average_rate = sim_metrics_vs_min_ue_rate_2serve_ue(
                f_env,
                f_channel,
                lf_planners,
                samp_int=samp_int,
                ue_speed=ue_speed,
                time_duration=time_duration,
                max_uav_speed=max_uav_speed,
                num_pts_ue_grid=num_pts_ue_grid,
                min_uav_rate=min_uav_rate,
                min_ue_rate=min_ue_rate,
                ind_mc_iter_start=ind_mc_iter_start,
                num_mc_iter=num_mc_iter,
                loc_bs=bs_loc,
                save_at=save_single_pt_at,
                b_run_new=b_run_new,
                b_return_failure_only=True)

            # num_min_ue_rate x num_planners x num_time_steps
            ldv_rate_vs_time.append(dv_ue_rates_vs_time)
            ld_time_to_min_ue_rate.append(d_time_to_min_ue_rate)
            ld_prob_failure.append(d_prob_failure)
            ldv_cumsum_data_over_time.append(dv_cumsum_data_over_time)
            ld_average_rate.append(d_average_rate)

        dl_time_to_min_ue_rate = ld_to_dl(ld_time_to_min_ue_rate)
        dl_prob_outage = None
        dl_prob_failure = ld_to_dl(ld_prob_failure)
        dl_average_rate = ld_to_dl(ld_average_rate)

        l_G = plot_performance(dl_time_to_min_ue_rate,
                               dl_prob_outage,
                               dl_prob_failure,
                               dl_average_rate,
                               ldv_rate_vs_time,
                               ldv_cumsum_data_over_time,
                               samp_int=samp_int,
                               xlabel='Min. UE rate [Mbps]',
                               xticks=v_min_ue_rate / 1e6,
                               l_min_ue_rate=list(v_min_ue_rate))

        G_1 = GFigure.concatenate([l_G[0], l_G[2]])
        # G_1.layout = "constrained"
        G_1.l_subplots[0].xlabel = ""
        G_1.l_subplots[1].num_legend_cols = 2
        G_1.l_subplots[1].legend_loc = "upper left"
        for curve in G_1.l_subplots[0].l_curves:
            curve.legend_str = ""

        return l_G + [G_1]

    # fraction of realizations requiring lifting vs dist_bs_ue for different
    # min_uav_rate
    def experiment_2_1007_15_15(l_args):

        exp_name = "exp_2_1007_15_15"
        save_at = './output/relay_placement_experiments_2/' + exp_name + '/'
        os.makedirs(save_at, exist_ok=True)

        b_run_new = True  # False

        ind_mc_iter_start = 0  # sim. for new envs starts from 5000
        num_mc_iter = 1  # 1500

        v_fraction_pts_disable = np.array([
            .0,
            .1,
            .2,
            .4,
            .6,
        ])

        loc_bs = np.array([10, 10, 0])  # [90, 420, 0]  [20, 470, 0.]
        loc_ue = np.array([470, 470, 0])
        # dist_bs_ue = 500

        min_ue_rate = 10e6  # 200e3

        # 42.5, 45., 47.5, 50., 52.5
        v_min_uav_rate = np.array([
            37.5,
            40.,
            42.5,
            45.,
            47.5,
        ]) * 1e6

        max_uav_speed = 10  # m/s
        samp_int = 2

        prpp_max_num_neighbors = 100
        prpp_num_nodes = 2000
        prpp_destination = "min_ue_rate"

        f_env = lambda: CustomBlockUrbanEnvironment(
            num_pts_slf_grid=[50, 50, 20],
            num_pts_fly_grid=[12, 12, 4],  # [12, 12, 5]
            min_fly_height=10,
            building_height=[20, 40],  # 60, 
            building_absorption=100)

        f_channel = lambda env: TomographicChannel(slf=env.slf,
                                                   freq_carrier=6e9,
                                                   tx_dbpower=watt_to_dbW(.05),
                                                   noise_dbpower=-97,
                                                   bandwidth=20e6,
                                                   min_link_capacity=2,
                                                   max_link_capacity=7,
                                                   antenna_dbgain_tx=12,
                                                   antenna_dbgain_rx=12)

        lf_planners = [
            lambda env, channel, min_uav_rate=min_uav_rate:
            RandomRoadmapPathPlanner(environment=env,
                                     channel=channel,
                                     num_uavs=2,
                                     num_nodes=prpp_num_nodes,
                                     max_num_neighbors=prpp_max_num_neighbors,
                                     min_uav_rate=min_uav_rate,
                                     mode_draw_conf_pt="feasible",
                                     mode_connect='min_rate_only',
                                     destination=prpp_destination,
                                     min_ue_rate=min_ue_rate,
                                     b_conf_pts_meet_min_ue_rate=False,
                                     b_tentative=True,
                                     name_custom=f'PRFI - {min_uav_rate/1e6}')
            for min_uav_rate in v_min_uav_rate
        ]

        lll_status = []

        for percent_pts_disable in v_fraction_pts_disable:
            print('-' * 20)
            print(f'Percentage of points to disable: {percent_pts_disable}')
            save_single_pt_at = save_at + f'percent{percent_pts_disable*100}' + '/'
            os.makedirs(save_single_pt_at, exist_ok=True)

            ll_is_optimal = sim_optimality_vs_dist_bs_static_ue(
                f_env,
                f_channel,
                lf_planners,
                percent_pts_disable=percent_pts_disable,
                samp_int=samp_int,
                max_uav_speed=max_uav_speed,
                ind_mc_iter_start=ind_mc_iter_start,
                num_mc_iter=num_mc_iter,
                min_ue_rate=min_ue_rate,
                loc_bs=loc_bs,
                loc_ue=loc_ue,
                save_at=save_single_pt_at,
                b_run_new=b_run_new)

            # num_percent x num_mc_iter x num_planners
            lll_status.append(ll_is_optimal)

        # num_planners x num_percent x num_mc_iter
        # 0: not optimal, 1: optimal, np.nan: failure
        t_status = np.array(lll_status).transpose(2, 0, 1)
        t_optimal = np.nan_to_num(t_status, nan=0)
        t_failures = np.where(np.isnan(t_status), 1, 0)

        # num_planners x num_percent
        m_optimal = np.sum(
            t_optimal, axis=-1) / (num_mc_iter - np.sum(t_failures, axis=-1))
        m_failure = np.mean(t_failures, axis=-1)

        F_gap = GFigure(
            # xlabel='Fraction of disabled points',
            ylabel='Prob. of Guar. Opt.')
        F_fail = GFigure(xlabel='Fraction of disabled points',
                         ylabel='Prob. of Infeas.')
        l_styles = ['-o', '-s', '-^', '-v', '-<', '->', '-*', '-x', '-D', '-P']
        for ind_fract, fraction in enumerate(v_min_uav_rate):
            F_gap.add_curve(
                xaxis=v_fraction_pts_disable,
                yaxis=m_optimal[ind_fract],
                styles=l_styles[ind_fract],
                # legend=f'Min. UE rate - {min_ue_rate/1e6} Mbps'
            )
            F_fail.add_curve(xaxis=v_fraction_pts_disable,
                             yaxis=m_failure[ind_fract],
                             styles=l_styles[ind_fract],
                             legend=f'Min. UE rate - {fraction/1e6} Mbps')

        S_gap = GFigure(
            # xlabel='Fraction of disabled points',
            ylabel='Prob. of Guar. Opt.',
            xticks=v_min_uav_rate / 1e6)
        S_fail = GFigure(xlabel='Min. UAV Rate [Mbps]',
                         ylabel='Prob. of Infeas.',
                         xticks=v_min_uav_rate / 1e6)

        # S_gap.
        # S_fail.xticks = v_min_uav_rate / 1e6

        l_styles = ['-o', '-s', '-^', '-v', '-<', '->', '-*', '-x', '-D', '-P']
        for ind_fract, fraction in enumerate(v_fraction_pts_disable):
            S_gap.add_curve(
                xaxis=v_min_uav_rate / 1e6,
                yaxis=m_optimal.T[ind_fract],
                styles=l_styles[ind_fract],
                # legend=f'Min. UE rate - {min_ue_rate/1e6} Mbps'
            )
            S_fail.add_curve(xaxis=v_min_uav_rate / 1e6,
                             yaxis=m_failure.T[ind_fract],
                             styles=l_styles[ind_fract],
                             legend=f'Frac. of disabled points - {fraction}')

        S = GFigure.concatenate([S_gap, S_fail])
        G = GFigure.concatenate([F_gap, F_fail])

        return [G, S]

    """
        70. Experiments to compare algorithms planning a path to serve a moving
            UE
    """

    # vs min_ue_rate: time to connect, frac. outage time, total transferred data
    def experiment_7000(l_args):
        """
        This experiment plots
            + Time to connect: Tm(min_ue_rate) vs. min_ue_rate, where
              Tm(min_ue_rate) is the MC average of the min time to reach a rate
              of min_ue_rate. Those MC iterations where this min rate is never
              achieved are not considered.
            
            + Frac. outage time: the MC average of the ratios of i) the number
              of time instants in which the user does not have min_ue_rate to
              ii) the total number of time instants.
            
            + Total transferred data: expected rate vs. time.
        """

        exp_name = "exp7000"
        save_at = './output/relay_placement_experiments/' + exp_name + '/'

        b_run_new = True
        ind_mc_iter_start = 0
        num_mc_iter = 1

        min_uav_rate = 200e3
        max_uav_speed = 7  # m/s
        samp_int = 60.237 / max_uav_speed
        ue_speed = 2
        num_pts_ue_grid = [25, 25, 1]
        time_duration = 300
        bs_loc = np.array([20, 470, 0.])

        # parameters for probabilistic roadmap path planners
        prpp_max_num_neighbors = 100
        prpp_num_nodes = 2000
        prpp_destination = "min_ue_rate"

        ldv_rate_vs_time = []
        ld_time_to_min_ue_rate = []
        ld_prob_outage = []
        ld_prob_failure = []
        ldv_cumsum_data_over_time = []
        ld_average_rate = []

        v_min_ue_rate = np.array([30, 50, 70, 90, 110, 130]) * 1e6

        f_env = lambda: CustomBlockUrbanEnvironment(
            num_pts_slf_grid=[50, 50, 20],
            num_pts_fly_grid=[12, 12, 8],
            min_fly_height=10,
            building_height=[20, 75],
            building_absorption=1)
        fly_height = f_env().fly_grid.list_pts()[:, 2].max()
        f_channel = lambda env: TomographicChannel(slf=env.slf,
                                                   freq_carrier=6e9,
                                                   tx_dbpower=watt_to_dbW(.05),
                                                   noise_dbpower=-97,
                                                   bandwidth=20e6,
                                                   min_link_capacity=2,
                                                   max_link_capacity=7,
                                                   antenna_dbgain_tx=12,
                                                   antenna_dbgain_rx=12)

        for min_ue_rate in v_min_ue_rate:
            print('-' * 20)
            print(f'\nMin_ue_rate: {min_ue_rate/1e6} Mbps')
            save_single_pt_at = save_at + f'minUeRate{int(min_ue_rate/1e6):003}' + '/'
            os.makedirs(save_single_pt_at, exist_ok=True)

            lf_planners = [
                lambda env, channel: SingleRelayMidpointPathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Zeng et al.'),
                #
                lambda env, channel: SingleRelayAlhouraniMaxRatePathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Ghazzai et al.'),
                #
                lambda env, channel: TwoRelaysOneMidOneAboveUEPathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Lee et al.'),
                #
                lambda env, channel: TwoRelaysMaxRatePathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Zhang et al.'),
                #
                lambda env, channel: UniformlySpreadRelaysPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Yanmaz et al.'),
                #
                lambda env, channel: SegmentationPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    max_num_neighbors=prpp_max_num_neighbors,
                    min_uav_rate=min_uav_rate,
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    num_known_ue_locs=17,
                    num_locs_to_replan=15,
                    name_custom='Benchmark 6'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=prpp_num_nodes,
                    max_num_neighbors=prpp_max_num_neighbors,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    ue_rate_below_target_penalty=1e6,
                    b_tentative=True,
                    name_custom='PRFI (Tentative)'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=prpp_num_nodes,
                    max_num_neighbors=prpp_max_num_neighbors,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    ue_rate_below_target_penalty=1e6,
                    b_tentative=False,
                    name_custom='PRFI'),
                #
            ]

            d_time_to_min_ue_rate, d_prob_failure, d_prob_outage, dv_ue_rates_vs_time, dv_cumsum_data_over_time, d_average_rate = sim_metrics_vs_min_ue_rate_2serve_ue(
                f_env,
                f_channel,
                lf_planners,
                samp_int=samp_int,
                ue_speed=ue_speed,
                max_uav_speed=max_uav_speed,
                num_pts_ue_grid=num_pts_ue_grid,
                time_duration=time_duration,
                min_uav_rate=min_uav_rate,
                min_ue_rate=min_ue_rate,
                ind_mc_iter_start=ind_mc_iter_start,
                num_mc_iter=num_mc_iter,
                loc_bs=bs_loc,
                save_at=save_single_pt_at,
                b_run_new=b_run_new)

            # num_min_ue_rate x num_planners x num_time_steps
            ldv_rate_vs_time.append(dv_ue_rates_vs_time)
            ld_time_to_min_ue_rate.append(d_time_to_min_ue_rate)
            ld_prob_outage.append(d_prob_outage)
            ld_prob_failure.append(d_prob_failure)
            ldv_cumsum_data_over_time.append(dv_cumsum_data_over_time)
            ld_average_rate.append(d_average_rate)

        dl_time_to_min_ue_rate = ld_to_dl(ld_time_to_min_ue_rate)
        dl_prob_outage = ld_to_dl(ld_prob_outage)
        dl_prob_failure = ld_to_dl(ld_prob_failure)
        dl_average_rate = ld_to_dl(ld_average_rate)

        return plot_performance(dl_time_to_min_ue_rate,
                                dl_prob_outage,
                                dl_prob_failure,
                                dl_average_rate,
                                ldv_rate_vs_time,
                                ldv_cumsum_data_over_time,
                                samp_int=samp_int,
                                xlabel='Min. UE rate [Mbps]',
                                xticks=v_min_ue_rate / 1e6,
                                l_min_ue_rate=list(v_min_ue_rate))

    # vs dist_bs_ue: time to connect, frac. outage time, total transferred data
    def experiment_7013(l_args):
        """
        This experiment plots
            + Time to connect: Tm(min_ue_rate) vs. dist_bs_ue, where
              Tm(min_ue_rate) is the MC average of the min time to reach a rate
              of min_ue_rate. Those MC iterations where this min rate is never
              achieved are not considered.
            
            + Frac. outage time: the MC average of the ratios of i) the number
              of time instants in which the user does not have min_ue_rate to
              ii) the total number of time instants.
            
            + Total transferred data: expected rate vs. time.
        
        """

        exp_name = "exp7013"
        save_at = './output/relay_placement_experiments/' + exp_name + '/'
        os.makedirs(save_at, exist_ok=True)

        b_run_new = True
        ind_mc_iter_start = 0
        num_mc_iter = 1

        min_uav_rate = 200e3
        max_uav_speed = 7  # m/s
        samp_int = 60.237 / max_uav_speed
        ue_speed = 2
        num_pts_ue_grid = [25, 25, 1]
        time_duration = 300
        bs_loc = np.array([20, 470, 0.])

        prpp_max_num_neighbors = 100
        prpp_num_nodes = 2000
        prpp_destination = "min_ue_rate"

        ldv_rate_vs_time = []
        ld_time_to_min_ue_rate = []
        ld_prob_outage = []
        ld_prob_failure = []
        ldv_cumsum_data_over_time = []
        ld_average_rate = []

        min_ue_rate = 110e6
        # [70, 100, 150, 200, 250, 300, 350, 400, 450]
        v_dist_bs_ue = np.array([70, 100, 150, 200, 250, 300, 350, 400, 450])

        f_env = lambda: CustomBlockUrbanEnvironment(
            num_pts_slf_grid=[50, 50, 20],
            num_pts_fly_grid=[12, 12, 8],
            min_fly_height=10,
            building_height=[20, 75],
            building_absorption=1)
        fly_height = f_env().fly_grid.list_pts()[:, 2].max()
        f_channel = lambda env: TomographicChannel(slf=env.slf,
                                                   freq_carrier=6e9,
                                                   tx_dbpower=watt_to_dbW(.05),
                                                   noise_dbpower=-97,
                                                   bandwidth=20e6,
                                                   min_link_capacity=2,
                                                   max_link_capacity=7,
                                                   antenna_dbgain_tx=12,
                                                   antenna_dbgain_rx=12)

        lf_planners = [
            lambda env, channel: SingleRelayMidpointPathPlanner(
                environment=env,
                channel=channel,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Zeng et al.'),
            #
            lambda env, channel: SingleRelayAlhouraniMaxRatePathPlanner(
                environment=env,
                channel=channel,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Ghazzai et al.'),
            #
            lambda env, channel: TwoRelaysOneMidOneAboveUEPathPlanner(
                environment=env,
                channel=channel,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Lee et al.'),
            #
            lambda env, channel: TwoRelaysMaxRatePathPlanner(
                environment=env,
                channel=channel,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Zhang et al.'),
            #
            lambda env, channel: UniformlySpreadRelaysPathPlanner(
                environment=env,
                channel=channel,
                num_uavs=2,
                min_uav_rate=min_uav_rate,
                fly_height=fly_height,
                name_custom='Yanmaz et al.'),
            #
            lambda env, channel: SegmentationPathPlanner(
                environment=env,
                channel=channel,
                num_uavs=2,
                max_num_neighbors=prpp_max_num_neighbors,
                min_uav_rate=min_uav_rate,
                mode_connect='min_rate_only',
                destination=prpp_destination,
                min_ue_rate=min_ue_rate,
                num_known_ue_locs=17,
                num_locs_to_replan=15,
                name_custom='Benchmark 6'),
            #
            lambda env, channel: RandomRoadmapPathPlanner(
                environment=env,
                channel=channel,
                num_uavs=2,
                num_nodes=prpp_num_nodes,
                max_num_neighbors=prpp_max_num_neighbors,
                min_uav_rate=min_uav_rate,
                mode_draw_conf_pt="feasible",
                mode_connect='min_rate_only',
                destination=prpp_destination,
                min_ue_rate=min_ue_rate,
                b_conf_pts_meet_min_ue_rate=False,
                ue_rate_below_target_penalty=1e6,
                b_tentative=True,
                name_custom='PRFI (Tentative)'),
            #
            lambda env, channel: RandomRoadmapPathPlanner(
                environment=env,
                channel=channel,
                num_uavs=2,
                num_nodes=prpp_num_nodes,
                max_num_neighbors=prpp_max_num_neighbors,
                min_uav_rate=min_uav_rate,
                mode_draw_conf_pt="feasible",
                mode_connect='min_rate_only',
                destination=prpp_destination,
                min_ue_rate=min_ue_rate,
                b_conf_pts_meet_min_ue_rate=False,
                ue_rate_below_target_penalty=1e6,
                b_tentative=False,
                name_custom='PRFI'),
            #
        ]

        for dist_bs_ue in v_dist_bs_ue:
            print('-' * 20)
            print(f'Distance BS - UE: {dist_bs_ue} m')
            save_single_pt_at = save_at + f'distBSUE{dist_bs_ue:003}' + '/'
            os.makedirs(save_single_pt_at, exist_ok=True)

            d_time_to_min_ue_rate, d_prob_failure, d_prob_outage, dv_ue_rates_vs_time, dv_cumsum_data_over_time, d_average_rate = sim_metrics_vs_min_ue_rate_2serve_ue(
                f_env,
                f_channel,
                lf_planners,
                samp_int=samp_int,
                ue_speed=ue_speed,
                num_pts_ue_grid=num_pts_ue_grid,
                max_uav_speed=max_uav_speed,
                time_duration=time_duration,
                min_uav_rate=min_uav_rate,
                min_ue_rate=min_ue_rate,
                ind_mc_iter_start=ind_mc_iter_start,
                num_mc_iter=num_mc_iter,
                loc_bs=bs_loc,
                dist_bs_ue=dist_bs_ue,
                save_at=save_single_pt_at,
                b_run_new=b_run_new)

            ldv_rate_vs_time.append(dv_ue_rates_vs_time)
            ld_time_to_min_ue_rate.append(d_time_to_min_ue_rate)
            ld_prob_outage.append(d_prob_outage)
            ld_prob_failure.append(d_prob_failure)
            ldv_cumsum_data_over_time.append(dv_cumsum_data_over_time)
            ld_average_rate.append(d_average_rate)

        dl_time_to_min_ue_rate = ld_to_dl(ld_time_to_min_ue_rate)
        dl_prob_outage = ld_to_dl(ld_prob_outage)
        dl_prob_failure = ld_to_dl(ld_prob_failure)
        dl_average_rate = ld_to_dl(ld_average_rate)

        l_G = plot_performance(dl_time_to_min_ue_rate,
                               dl_prob_outage,
                               dl_prob_failure,
                               dl_average_rate,
                               ldv_rate_vs_time,
                               ldv_cumsum_data_over_time,
                               samp_int=samp_int,
                               xlabel="Distance BS-UE [m]",
                               xticks=v_dist_bs_ue,
                               l_min_ue_rate=min_ue_rate)

        G_1 = GFigure.concatenate([l_G[1], l_G[2]])
        G_1.l_subplots[0].xlabel = ""
        G_1.l_subplots[1].num_legend_cols = 2
        G_1.l_subplots[1].legend_loc = "lower right"
        for curve in G_1.l_subplots[0].l_curves:
            curve.legend_str = ""
        G_1.l_subplots[1].l_curves[-3].legend_str = "Benchmark 6"

        return l_G + [G_1]

    # PRFI only; vs min ue rate: time to connect, frac. outage time, total
    # transferred data
    def experiment_7027(l_args):
        """
        This experiment plots
            + Time to connect: Tm(min_ue_rate) vs. num. conf pts, where
              Tm(min_ue_rate) is the MC average of the min time to reach a rate
              of min_ue_rate. Those MC iterations where this min rate is never
              achieved are not considered.
            
            + Frac. outage time: the MC average of the ratios of i) the number
              of time instants in which the user does not have min_ue_rate to
              ii) the total number of time instants.
            
            + Total transferred data: expected rate vs. time.
        
        """

        exp_name = "exp7027"
        save_at = './output/relay_placement_experiments/' + exp_name + '/'
        os.makedirs(save_at, exist_ok=True)

        b_run_new = True

        ind_mc_iter_start = 0
        num_mc_iter = 1

        min_uav_rate = 200e3
        max_uav_speed = 7  # m/s
        samp_int = 60.237 / max_uav_speed
        ue_speed = 2
        num_pts_ue_grid = [25, 25, 1]
        time_duration = 300
        bs_loc = np.array([20, 470, 0.])

        # parameters for probabilistic roadmap path planners
        prpp_destination = "min_ue_rate"

        ldv_rate_vs_time = []
        ld_time_to_min_ue_rate = []
        ld_prob_outage = []
        ld_prob_failure = []
        ldv_cumsum_data_over_time = []
        ld_average_rate = []

        # [10, 30, 50, 70, 90, 110, 130]
        v_min_ue_rate = np.array([10, 30, 50, 70, 90, 110, 130]) * 1e6

        f_env = lambda: CustomBlockUrbanEnvironment(
            num_pts_slf_grid=[50, 50, 20],
            num_pts_fly_grid=[12, 12, 8],
            min_fly_height=10,
            building_height=[20, 75],
            building_absorption=1e6)
        f_channel = lambda env: TomographicChannel(slf=env.slf,
                                                   freq_carrier=6e9,
                                                   tx_dbpower=watt_to_dbW(.05),
                                                   noise_dbpower=-97,
                                                   bandwidth=20e6,
                                                   min_link_capacity=2,
                                                   max_link_capacity=7,
                                                   antenna_dbgain_tx=12,
                                                   antenna_dbgain_rx=12)

        for min_ue_rate in v_min_ue_rate:
            print('-' * 20)
            print(f'\nMin_ue_rate: {min_ue_rate}')
            save_single_pt_at = save_at + f'minUeRate{int(min_ue_rate/1e6):003}' + '/'
            os.makedirs(save_single_pt_at, exist_ok=True)

            lf_planners = [
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=100,
                    max_num_neighbors=100,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    ue_rate_below_target_penalty=1e6,
                    b_tentative=True,
                    name_custom='PRFI (Tentative)'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=100,
                    max_num_neighbors=100,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    ue_rate_below_target_penalty=1e6,
                    b_tentative=False,
                    name_custom='PRFI (100 nodes)'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=500,
                    max_num_neighbors=100,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    ue_rate_below_target_penalty=1e6,
                    b_tentative=False,
                    name_custom='PRFI (500 nodes)'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=1000,
                    max_num_neighbors=100,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    ue_rate_below_target_penalty=1e6,
                    b_tentative=False,
                    name_custom='PRFI (1000 nodes)'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=2000,
                    max_num_neighbors=100,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    ue_rate_below_target_penalty=1e6,
                    b_tentative=False,
                    name_custom='PRFI (2000 nodes)')
            ]

            d_time_to_min_ue_rate, d_prob_failure, d_prob_outage, dv_ue_rates_vs_time, dv_cumsum_data_over_time, d_average_rate = sim_metrics_vs_min_ue_rate_2serve_ue(
                f_env,
                f_channel,
                lf_planners,
                samp_int=samp_int,
                ue_speed=ue_speed,
                num_pts_ue_grid=num_pts_ue_grid,
                max_uav_speed=max_uav_speed,
                time_duration=time_duration,
                min_uav_rate=min_uav_rate,
                min_ue_rate=min_ue_rate,
                ind_mc_iter_start=ind_mc_iter_start,
                num_mc_iter=num_mc_iter,
                loc_bs=bs_loc,
                save_at=save_single_pt_at,
                b_run_new=b_run_new)

            ldv_rate_vs_time.append(dv_ue_rates_vs_time)
            ld_time_to_min_ue_rate.append(d_time_to_min_ue_rate)
            ld_prob_outage.append(d_prob_outage)
            ld_prob_failure.append(d_prob_failure)
            ldv_cumsum_data_over_time.append(dv_cumsum_data_over_time)
            ld_average_rate.append(d_average_rate)

        dl_time_to_min_ue_rate = ld_to_dl(ld_time_to_min_ue_rate)
        dl_prob_outage = ld_to_dl(ld_prob_outage)
        dl_prob_failure = ld_to_dl(ld_prob_failure)
        dl_average_rate = ld_to_dl(ld_average_rate)

        plot_test = False
        if plot_test:
            print('Plot test')
            l_num_nodes = [0, 500, 1000, 2000]
            l_ind_rate = [0, 1, 2, 3]
            ll_avg_rate = []
            for ind_rate in l_ind_rate:
                l_avg_rate = []
                for ind, key in enumerate(dl_average_rate.keys()):
                    l_avg_rate.append(dl_average_rate[key][ind_rate])
                ll_avg_rate.append(l_avg_rate)

            G_avg_rate = GFigure(xlabel="Number of Conf. Points",
                                 ylabel="Average Rate [Mbps]",
                                 xticks=l_num_nodes)
            for ind_rate, avg_rate in enumerate(ll_avg_rate):
                G_avg_rate.add_curve(
                    xaxis=l_num_nodes,
                    yaxis=np.array(avg_rate) / 1e6,
                    legend=
                    f'Min. UE rate: {v_min_ue_rate[l_ind_rate[ind_rate]]/1e6} Mbps'
                )

            return G_avg_rate

        else:
            return plot_performance(dl_time_to_min_ue_rate,
                                    dl_prob_outage,
                                    dl_prob_failure,
                                    dl_average_rate,
                                    ldv_rate_vs_time,
                                    ldv_cumsum_data_over_time,
                                    samp_int=samp_int,
                                    xlabel="Min. UE rate [Mbps]",
                                    xticks=v_min_ue_rate / 1e6,
                                    l_min_ue_rate=list(v_min_ue_rate))

    # vs building heights: time to connect, frac. outage time, total transferred
    # data
    def experiment_7052(l_args):
        """
        This experiment plots
            + Time to connect: Tm(min_ue_rate) vs. building heights, where
              Tm(min_ue_rate) is the MC average of the min time to reach a rate
              of min_ue_rate. Those MC iterations where this min rate is never
              achieved are not considered.
            
            + Frac. outage time: the MC average of the ratios of i) the number
              of time instants in which the user does not have min_ue_rate to
              ii) the total number of time instants.
            
            + Total transferred data: expected rate vs. time.
        
        """

        exp_name = "exp7052"
        save_at = './output/relay_placement_experiments/' + exp_name + '/'
        os.makedirs(save_at, exist_ok=True)

        b_run_new = True
        ind_mc_iter_start = 0
        num_mc_iter = 1

        min_uav_rate = 200e3
        # distance btw two adjacent diagonal grid points: 14.678
        # np.sqrt(np.sum(np.square(env.fly_grid.spacing)))
        max_uav_speed = 7  # m/s
        samp_int = 60.237 / max_uav_speed
        ue_speed = 2
        num_pts_ue_grid = [25, 25, 1]
        time_duration = 300
        bs_loc = np.array([20, 470, 0.])

        prpp_max_num_neighbors = 100
        prpp_num_nodes = 2000
        prpp_destination = "min_ue_rate"

        ldv_rate_vs_time = []
        ld_time_to_min_ue_rate = []
        ld_prob_outage = []
        ld_prob_failure = []
        ldv_cumsum_data_over_time = []
        ld_average_rate = []

        min_ue_rate = 100e6

        dist_bs_ue = 450

        l_min_heights = [20, 30, 40, 50, 60]  # 20, 30, 40, 50, 60
        l_building_heights = [[height - 20] + [height + 20]
                              for height in l_min_heights]
        l_mean_building_heights = [
            int(np.sum(building_heights) / 2)
            for building_heights in l_building_heights
        ]

        f_channel = lambda env: TomographicChannel(slf=env.slf,
                                                   freq_carrier=6e9,
                                                   tx_dbpower=watt_to_dbW(.05),
                                                   noise_dbpower=-97,
                                                   bandwidth=20e6,
                                                   min_link_capacity=2,
                                                   max_link_capacity=7,
                                                   antenna_dbgain_tx=12,
                                                   antenna_dbgain_rx=12)

        for ind, building_heights in enumerate(l_building_heights):
            print('-' * 20)
            print(f'Mean building height: {l_mean_building_heights[ind]:02} m')
            save_single_pt_at = save_at + f'build{l_mean_building_heights[ind]:02}' + '/'
            os.makedirs(save_single_pt_at, exist_ok=True)

            f_env = lambda: CustomBlockUrbanEnvironment(
                num_pts_slf_grid=[50, 50, 20],
                num_pts_fly_grid=[12, 12, 8],
                min_fly_height=10,
                building_height=building_heights,
                building_absorption=1)
            fly_height = f_env().fly_grid.list_pts()[:, 2].max()

            lf_planners = [
                lambda env, channel: SingleRelayMidpointPathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Zeng et al.'),
                #
                lambda env, channel: SingleRelayAlhouraniMaxRatePathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Ghazzai et al.'),
                #
                lambda env, channel: TwoRelaysOneMidOneAboveUEPathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Lee et al.'),
                #
                lambda env, channel: TwoRelaysMaxRatePathPlanner(
                    environment=env,
                    channel=channel,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Zhang et al.'),
                #
                lambda env, channel: UniformlySpreadRelaysPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    min_uav_rate=min_uav_rate,
                    fly_height=fly_height,
                    name_custom='Yanmaz et al.'),
                #
                lambda env, channel: SegmentationPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    max_num_neighbors=prpp_max_num_neighbors,
                    min_uav_rate=min_uav_rate,
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    num_known_ue_locs=17,
                    num_locs_to_replan=15,
                    name_custom='Benchmark 6'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=prpp_num_nodes,
                    max_num_neighbors=prpp_max_num_neighbors,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    ue_rate_below_target_penalty=1e6,
                    b_tentative=True,
                    name_custom='PRFI (Tentative)'),
                #
                lambda env, channel: RandomRoadmapPathPlanner(
                    environment=env,
                    channel=channel,
                    num_uavs=2,
                    num_nodes=prpp_num_nodes,
                    max_num_neighbors=prpp_max_num_neighbors,
                    min_uav_rate=min_uav_rate,
                    mode_draw_conf_pt="feasible",
                    mode_connect='min_rate_only',
                    destination=prpp_destination,
                    min_ue_rate=min_ue_rate,
                    b_conf_pts_meet_min_ue_rate=False,
                    ue_rate_below_target_penalty=1e6,
                    b_tentative=False,
                    name_custom='PRFI'),
                #
            ]

            d_time_to_min_ue_rate, d_prob_failure, d_prob_outage, dv_ue_rates_vs_time, dv_cumsum_data_over_time, d_average_rate = sim_metrics_vs_min_ue_rate_2serve_ue(
                f_env,
                f_channel,
                lf_planners,
                samp_int=samp_int,
                ue_speed=ue_speed,
                num_pts_ue_grid=num_pts_ue_grid,
                max_uav_speed=max_uav_speed,
                time_duration=time_duration,
                min_uav_rate=min_uav_rate,
                min_ue_rate=min_ue_rate,
                ind_mc_iter_start=ind_mc_iter_start,
                num_mc_iter=num_mc_iter,
                loc_bs=bs_loc,
                dist_bs_ue=dist_bs_ue,
                save_at=save_single_pt_at,
                b_run_new=b_run_new)

            ldv_rate_vs_time.append(dv_ue_rates_vs_time)
            ld_time_to_min_ue_rate.append(d_time_to_min_ue_rate)
            ld_prob_outage.append(d_prob_outage)
            ld_prob_failure.append(d_prob_failure)
            ldv_cumsum_data_over_time.append(dv_cumsum_data_over_time)
            ld_average_rate.append(d_average_rate)

        dl_time_to_min_ue_rate = ld_to_dl(ld_time_to_min_ue_rate)
        dl_prob_outage = ld_to_dl(ld_prob_outage)
        dl_prob_failure = ld_to_dl(ld_prob_failure)
        dl_average_rate = ld_to_dl(ld_average_rate)

        return plot_performance(dl_time_to_min_ue_rate,
                                dl_prob_outage,
                                dl_prob_failure,
                                dl_average_rate,
                                ldv_rate_vs_time,
                                ldv_cumsum_data_over_time,
                                samp_int=samp_int,
                                xlabel="Mean building heights [m]",
                                xticks=l_mean_building_heights,
                                l_min_ue_rate=min_ue_rate)


class CustomBlockUrbanEnvironment(BlockUrbanEnvironment):

    num_roads_one_axis = 6
    num_buildings_one_axis = num_roads_one_axis - 1
    road_width = 40
    area_length_horizontal = 500
    area_height = 100
    area_len = [area_length_horizontal, area_length_horizontal, area_height]
    building_width = (area_length_horizontal -
                      num_roads_one_axis * road_width) / num_buildings_one_axis
    lower_bound_block_limits = np.arange(
        road_width - road_width / 2, area_length_horizontal - road_width / 2,
        road_width + building_width)
    upper_bound_block_limits = np.arange(
        road_width + building_width - road_width / 2,
        area_length_horizontal + building_width - road_width / 2,
        road_width + building_width)
    block_limits_x = np.array(
        [lower_bound_block_limits, upper_bound_block_limits]).T
    block_limits_y = block_limits_x


def plot_performance(dl_time_to_min_ue_rate,
                     dl_prob_outage,
                     dl_prob_failure,
                     dl_average_rate,
                     ldv_rate_vs_time,
                     ldv_cumsum_data_over_time,
                     samp_int=None,
                     xlabel=None,
                     xticks=None,
                     l_min_ue_rate=None):
    """ 
    Args:
        + ldv_rate_vs_time: num_params x num_planners x num_time_steps, where num_params is the number of varied parameters on the x axis.

    Returns:
        G_time_to_connect, G_avg_rate, G_outage, G_failure, *l_G_rate_vs_time, *l_G_cumsum_data_over_time            
    """

    assert samp_int is not None
    assert xlabel is not None
    assert xticks is not None
    assert l_min_ue_rate is not None

    if type(l_min_ue_rate) != list:
        l_min_ue_rate = [l_min_ue_rate] * len(xticks)

    G_time_to_connect = GFigure(xlabel=xlabel,
                                ylabel="Mean Time [s]",
                                xticks=xticks)
    G_avg_rate = GFigure(xlabel=xlabel,
                         ylabel="Average Rate [Mbps]",
                         xticks=xticks)

    G_failure = GFigure(xlabel=xlabel,
                        ylabel="Prob. of Failure",
                        xticks=xticks)

    ldv_ue_rates = ldv_rate_vs_time
    ldv_cum_sum_data_transfered_over_time = ldv_cumsum_data_over_time

    l_G_rate_vs_time = [
        GFigure(xlabel="Time [s]", ylabel="Mean UE Rate [Mbps]")
        for _ in range(len(ldv_rate_vs_time))
    ]
    l_G_cumsum_data_over_time = [
        GFigure(xlabel="Time [s]", ylabel="Total transferred data [Gb]")
        for _ in range(len(ldv_rate_vs_time))
    ]

    if len(dl_time_to_min_ue_rate.keys()) > 7:
        l_styles = [
            '-p#1f77b4', '-*#ff7f0e', '-^#2ca02c', '-v#8c564b', '-X#e377c2',
            '-d#9467bd', '-s#d62728', '-o#000000'
        ]
        l_styles_rate_vs_time = [
            '-#1f77b4', '-#ff7f0e', '-#2ca02c', '-#8c564b', '-#e377c2',
            '-#9467bd', '-#d62728', '-#000000'
        ]
    else:
        l_styles = [
            '-p#1f77b4',
            '-*#ff7f0e',
            '-^#2ca02c',
            '-v#8c564b',  # '-v#8c564b'
            '-X#e377c2',  # 'e377c2'
            '-s#d62728',  # '-s#17becf'
            '--o#000000'  # '-s#9467bd'
        ]
        l_styles_rate_vs_time = [
            '-#1f77b4', '-#ff7f0e', '-#2ca02c', '-#8c564b', '-#e377c2',
            '-#d62728', '-#000000'
        ]

    # plot min ue rate
    for ind_G, G in enumerate(l_G_rate_vs_time):
        num_time_steps = np.max(
            [len(l_rate) for l_rate in list(ldv_rate_vs_time[ind_G].values())])
        yaxis = [l_min_ue_rate[ind_G] / 1e6] * num_time_steps

        xaxis = np.linspace(0, num_time_steps, num_time_steps) * samp_int

        G.add_curve(xaxis=xaxis,
                    yaxis=yaxis,
                    styles=":k",
                    legend="Min. UE Rate")

    for ind, key in enumerate(dl_time_to_min_ue_rate.keys()):
        G_time_to_connect.add_curve(xaxis=xticks,
                                    yaxis=dl_time_to_min_ue_rate[key],
                                    styles=l_styles[ind],
                                    legend=key)

        G_avg_rate.add_curve(xaxis=xticks,
                             yaxis=np.array(dl_average_rate[key]) / 1e6,
                             styles=l_styles[ind],
                             legend=key)
        G_failure.add_curve(xaxis=xticks,
                            yaxis=dl_prob_failure[key],
                            styles=l_styles[ind],
                            legend=key)

        # plot mean ue rate and cumsum data vs time
        for ind_G, G in enumerate(l_G_rate_vs_time):
            yaxis_rate = ldv_ue_rates[ind_G][key] / 1e6
            num_time_steps = yaxis_rate.shape[0]
            x_time_steps = np.linspace(0, num_time_steps,
                                       num_time_steps) * samp_int

            # impose an upper bound on the rate for better illustration of
            # attaining min_ue_rate
            l_G_rate_vs_time[ind_G].add_curve(
                xaxis=x_time_steps,
                # yaxis=np.minimum(ldv_ue_rates[ind_G][key],
                #                  l_min_ue_rate[ind_G]) / 1e6,
                yaxis=ldv_ue_rates[ind_G][key] / 1e6,
                styles=l_styles_rate_vs_time[ind],
                legend=key)

            l_G_cumsum_data_over_time[ind_G].add_curve(
                xaxis=x_time_steps,
                yaxis=ldv_cum_sum_data_transfered_over_time[ind_G][key] / 1e9,
                styles=l_styles_rate_vs_time[ind],
                legend=key)

    if dl_prob_outage is not None:
        G_outage = GFigure(xlabel=xlabel,
                           ylabel="Prob. of Outage",
                           xticks=xticks)

        for ind, key in enumerate(dl_time_to_min_ue_rate.keys()):

            G_outage.add_curve(xaxis=xticks,
                               yaxis=dl_prob_outage[key],
                               styles=l_styles[ind],
                               legend=key)

        return [
            G_time_to_connect, G_avg_rate, G_outage, G_failure,
            *l_G_rate_vs_time, *l_G_cumsum_data_over_time
        ]
    else:

        return [
            G_time_to_connect, G_avg_rate, G_failure, *l_G_rate_vs_time,
            *l_G_cumsum_data_over_time
        ]


def ld_to_dl(ld):
    dl = dict()
    for d in ld:
        for key in d.keys():
            if key not in dl.keys():
                dl[key] = []
            dl[key].append(d[key])
    return dl
