import numpy as np
from relays.path_planners import PathPlanner

from common.grid import RectangularGrid3D
import pickle, os

# Functions to save and load realizations and results


def save_realization(mc_ind,
                     environment,
                     channel,
                     loc_bs,
                     lm_ue_path,
                     path_save_to=None):
    """
    Args:
        + loc_bs: (3,) array presents the location of the bs.

        + lm_ue_path: list of (1 x 3) array presents the path of the ue.        

    Store environment, channel, loc_bs, lm_ue_path into a pickle file and save at path_save_to.
    """
    assert path_save_to is not None

    d_realization = {
        "environment": environment,
        "channel": channel,
        "loc_bs": loc_bs,
        "lm_ue_path": lm_ue_path
    }

    path_save_to = path_save_to + f"mc{mc_ind:03}.pickle"
    # save to pickle file
    with open(path_save_to, 'wb') as file:
        pickle.dump(d_realization, file)


def save_or_load_results(ind_mc,
                         planner,
                         lm_path,
                         v_ue_rate,
                         path_save_to=None,
                         b_save=True):
    """
    If b_save is
        - True, store lm_path, v_ue_rate into a pickle file and save at
          path_save_to.

        - False, returns:
            + v_ue_rate: (num_samples, ) array presents the rate of the ue over
              time.
    """

    assert path_save_to is not None
    path_save_to = path_save_to + planner.name

    if b_save:

        os.makedirs(path_save_to, exist_ok=True)
        path_save_to += f"/mc{ind_mc:03}_output.pickle"

        d_results = {
            "lm_path": lm_path,
            "v_ue_rate": v_ue_rate,
        }

        # save to pickle file
        with open(path_save_to, 'wb') as file:
            pickle.dump(d_results, file)
    else:
        # load results
        path_load_from = path_save_to + f"/mc{ind_mc:03}_output.pickle"
        with open(path_load_from, 'rb') as file:
            d_results = pickle.load(file)

        return d_results["v_ue_rate"]


# move to BlockUrbanEnvironment


def is_out_buildings(env, pt, padding=5):
    # Padding around buildings to avoid the noncoincide between the buildings and slf grid
    v_safety = padding * np.array([[-1, 1]])
    block_limits_x_w_padding = env.block_limits_x + v_safety
    block_limits_y_w_padding = env.block_limits_y + v_safety

    if (np.sum(block_limits_x_w_padding - pt[0] >= 0) % 2
            == 1) and (np.sum(block_limits_y_w_padding - pt[1] >= 0) % 2 == 1):
        return False
    else:
        return True


def get_random_ue_loc(env,
                      channel,
                      min_ue_rate=None,
                      dist_bs_ue=None,
                      loc_bs=None,
                      num_max_tries=1000):
    """
    Generate a random UE location with distance dist_bs_ue to the BS, such that
    the initial rate between the BS and UE is lower than min_ue_rate.
    """

    def random_pts_w_dist(bs_loc):
        """
          Generate a random point on the circle with radius dist_bs_ue and
        center at bs_loc
        """
        theta = np.random.uniform(0, 2 * np.pi)

        ue_x = bs_loc[0] + dist_bs_ue * np.cos(theta)
        ue_y = bs_loc[1] + dist_bs_ue * np.sin(theta)

        return np.array([ue_x, ue_y, 0])

    def check_in_area_len(pt):

        is_in_x_limits = (0 <= pt[0]) and (pt[0] <= env.area_len[0] * 0.95)
        is_in_y_limits = (0 <= pt[1]) and (pt[1] <= env.area_len[1] * 0.95)

        if is_in_x_limits and is_in_y_limits:
            return True
        else:
            return False

    assert loc_bs is not None

    b_in_area_len = False

    remaining_tries = num_max_tries
    while (not b_in_area_len) and (remaining_tries > 0):
        # print('--- Regenerate ue_loc due to not on street')
        remaining_tries -= 1
        loc_ue = random_pts_w_dist(loc_bs)
        if check_in_area_len(loc_ue) and is_out_buildings(env, loc_ue):
            b_in_area_len = True
    assert remaining_tries > 0, f'Tried {str(num_max_tries)} times, cannot find a valid ue_loc'

    if channel.dbgain_to_capacity(channel.dbgain(loc_bs,
                                                 loc_ue)) >= min_ue_rate:
        # print('--- Regenerate ue_loc due to ininital rate >= min_ue_rate')
        return get_random_ue_loc(env,
                                 channel=channel,
                                 min_ue_rate=min_ue_rate,
                                 dist_bs_ue=dist_bs_ue,
                                 loc_bs=loc_bs)

    else:

        if np.all(loc_bs == None):
            return [loc_bs, loc_ue]
        else:
            return loc_ue


def gen_path_on_street(env,
                       channel,
                       min_ue_rate=None,
                       samp_int=None,
                       time_duration=None,
                       ue_speed=None,
                       num_pts_grid=None,
                       dist_bs_ue=None,
                       loc_bs=None,
                       debug=0):
    """
    Generate a random path on the street.

    Returns: 
        + lm_ue_path: list of (1,3) matrices of UE locations. lm_ue_path[n]
            is the UE location at time n*samp_int.
    
    """

    assert loc_bs is not None
    assert dist_bs_ue is not None

    # generate start location of the UE
    loc_ue = get_random_ue_loc(env,
                               channel=channel,
                               min_ue_rate=min_ue_rate,
                               dist_bs_ue=dist_bs_ue,
                               loc_bs=loc_bs)

    # static UE
    num_samples_total = int(np.ceil(time_duration / samp_int))
    if num_samples_total == 1:
        return loc_ue

    if num_pts_grid is None:
        lm_ue_path = [loc_ue[None, ...]] * num_samples_total
        return lm_ue_path

    ue_grid = RectangularGrid3D(area_len=env.area_len, num_pts=num_pts_grid)
    ue_grid.disable_by_indicator(env.building_indicator)

    m_ue_grid_pts = ue_grid.list_pts()

    num_nodes = m_ue_grid_pts.shape[0]

    # construct m_cost
    m_cost = np.full((num_nodes, num_nodes), np.nan)
    l_lines = []
    for ind_1 in range(num_nodes):
        for ind_2 in range(ind_1, num_nodes):
            # check if adjacent
            if ue_grid.are_adjacent(m_ue_grid_pts[ind_1],
                                    m_ue_grid_pts[ind_2]):
                m_cost[ind_1, ind_2] = np.linalg.norm(m_ue_grid_pts[ind_1] -
                                                      m_ue_grid_pts[ind_2])
                m_cost[ind_2, ind_1] = m_cost[ind_1, ind_2]
                l_lines.append([m_ue_grid_pts[ind_1], m_ue_grid_pts[ind_2]])

    ind_node_start = ue_grid.nearest_pt_ind(loc_ue)[1]

    lm_ue_path = []
    num_samples_so_far = 0
    while num_samples_so_far < num_samples_total:

        # Choose the destination
        ind_node_dest = np.random.randint(0, num_nodes)
        if ind_node_dest == ind_node_start:
            continue

        # Find path towards the destination
        l_ind_path_ue = PathPlanner.get_shortest_path(
            m_cost, ind_node_start=ind_node_start, ind_nodes_end=ind_node_dest)
        lm_path = [
            m_ue_grid_pts[ind][np.newaxis, ...] for ind in l_ind_path_ue
        ]
        lm_path = list(PathPlanner.resample(lm_path, samp_int, ue_speed))

        num_samples_to_add = min(len(lm_path),
                                 num_samples_total - num_samples_so_far)
        lm_ue_path += lm_path[:num_samples_to_add]
        num_samples_so_far += num_samples_to_add

        ind_node_start = ind_node_dest

    if debug > 0:
        env.l_users = [ue_loc for ue_loc in m_ue_grid_pts]
        env.l_lines = l_lines
        env.plot_path(lm_ue_path)

    return lm_ue_path


def get_time_to_min_ue_rate(v_ue_rate, min_ue_rate, samp_int):
    """
    Returns
        - `time_to_min_ue_rate`: n * samp_int, where n is the smallest integer
          such that v_ue_rate[n] >= min_ue_rate. If such n does not exist,
          returns None.
    """
    for ind_rate in range(len(v_ue_rate)):
        if v_ue_rate[ind_rate] >= min_ue_rate:
            return ind_rate * samp_int
    return None


def single_mc_rate(l_planners,
                   channel,
                   min_uav_rate=None,
                   samp_int=None,
                   loc_bs=None,
                   lm_ue_path=None,
                   max_uav_speed=None,
                   num_samples=None,
                   ind_mc=None,
                   path_save_to=None,
                   b_run_new=True):
    """
    Args:
        - `b_run_new`: if True, generate a new realization, plan path, and
          compute rate; else load saved results.
        
    Returns: 
        - `l_ue_rates_vs_time`: list whose n-th entry is the rate vs. time of
          l_planners[n].
    """

    lv_ue_rates_vs_time = []
    for planner in l_planners:

        if b_run_new:
            # moving ue
            if type(lm_ue_path) is list:
                lm_path = planner.plan_path_to_serve_moving_ue(
                    loc_bs, lm_ue_path)
            # static ue
            else:
                lm_path = planner.plan_path_to_serve_static_ue(
                    bs_loc=loc_bs,
                    ue_loc=lm_ue_path,
                    samp_int=samp_int,  #s
                    max_uav_speed=max_uav_speed)

            # found a path
            if lm_path is not None:
                v_ue_rate = PathPlanner.rate_from_path(
                    channel,
                    lm_path,
                    min_uav_rate,
                    loc_bs,
                    lm_ue_path=lm_ue_path)[1]
            # zero rates
            else:
                if num_samples is None:
                    v_ue_rate = np.array([])
                else:
                    v_ue_rate = np.array([0] * num_samples)
            # save results
            if path_save_to is not None:
                save_or_load_results(ind_mc, planner, lm_path, v_ue_rate,
                                     path_save_to)
        # load results
        else:
            v_ue_rate = save_or_load_results(ind_mc,
                                             planner,
                                             None,
                                             None,
                                             path_save_to,
                                             b_save=b_run_new)

        lv_ue_rates_vs_time.append(v_ue_rate)

    return lv_ue_rates_vs_time


def get_mean_time_to_min_ue_rate_n_prob(samp_int,
                                        l_planners,
                                        llv_ue_rates_vs_time=None,
                                        min_ue_rate=None,
                                        b_return_failure_only=True):
    """
    Args:
        + llv_ue_rates_vs_time: (num_mc_iter x number_planners x
          num_time_samples)

    If `b_return_failure_only` is True, returns:
        1. d_time_to_min_ue_rate: keys are names of planners, values are the
           sample means of the time to obtain min_ue_rate (time to connect) over
           realizations in which the planner attains min_ue_rate.
        
        2. d_prob_of_failure: keys are names of planners, values are the ratios
          of i) the number of realizations in which the planner fails to attain
          min_ue_rate to ii) the total number of realizations.        

    Else returns:
        + d_time_to_min_ue_rate: as defined in (1).

        + d_prob_of_outage: keys are names of planners, values are the sample
            means over realizations of the outage probabilities defined as the
            ratio of i) the number of time instants in which the user does not
            have min_ue_rate to ii) the total number of time instants.
    """

    num_planners = len(l_planners)

    # obtain ll_times_to_min_ue_rate from llv_ue_rates_vs_time
    # ll_times_to_min_ue_rate: (num_mc_iter x number_planners).
    ll_times_to_min_ue_rate = [[
        get_time_to_min_ue_rate(v_ue_rate, min_ue_rate, samp_int)
        for v_ue_rate in lv_ue_rates_vs_time
    ] for lv_ue_rates_vs_time in llv_ue_rates_vs_time]

    # num_path_planners x num_mc
    m_times_to_min_ue_rate = np.array(ll_times_to_min_ue_rate).T
    m_failures = (m_times_to_min_ue_rate == None)
    if np.sum(m_failures == True) != 0:
        m_times_to_min_ue_rate[m_failures] = None

    v_num_successes = np.empty(num_planners)
    v_num_successes[:] = np.nan
    v_mean_time_to_min_ue_rate = np.copy(v_num_successes)

    for ind_planner in range(num_planners):
        v_b_success = (m_times_to_min_ue_rate[ind_planner]
                       != None) * (m_times_to_min_ue_rate[ind_planner] != 0)
        v_num_successes[ind_planner] = np.sum(v_b_success)
        if v_num_successes[ind_planner] == 0:
            v_mean_time_to_min_ue_rate[ind_planner] = np.nan
        else:
            v_mean_time_to_min_ue_rate[ind_planner] = np.sum(
                m_times_to_min_ue_rate[ind_planner][
                    v_b_success == True]) / v_num_successes[ind_planner]

    v_prob_of_failure = np.mean(m_failures, axis=1)

    d_time_to_min_ue_rate = {
        str(l_planners[ind_pp]): v_mean_time_to_min_ue_rate[ind_pp]
        for ind_pp in range(num_planners)
    }

    d_prob_failure = {
        str(l_planners[ind_pp]): v_prob_of_failure[ind_pp]
        for ind_pp in range(num_planners)
    }

    if b_return_failure_only:
        return d_time_to_min_ue_rate, d_prob_failure, None

    else:

        num_time_steps = llv_ue_rates_vs_time[0][0].shape[0]
        # num_mc x num_planners x num_time_samples
        m_ue_rates_vs_time = np.array(llv_ue_rates_vs_time)
        v_mean_outage = np.mean(
            np.sum(m_ue_rates_vs_time < min_ue_rate, axis=2) / num_time_steps,
            axis=0)

        d_prob_outage = {
            str(l_planners[ind_pp]): v_mean_outage[ind_pp]
            for ind_pp in range(num_planners)
        }

        return d_time_to_min_ue_rate, d_prob_failure, d_prob_outage


def get_ue_rates_vs_time(l_planners, llv_ue_rates_vs_time, samp_int=None):
    """
    Average the ue_rates_vs_time over the realizations in llv_ue_rates_vs_time.

    Args:
        - `llv_ue_rates_vs_time`: (num_mc x num_planners x num_time_samples).
    
    If `samp_int` is 
        + None, returns dv_ue_rates_vs_time: keys are names of planners, values are the rate vs. time

        + not None, returns: 
            1. dv_ue_rates_vs_time: keys are names of planners, values are vectors of rate vs. time

            2. dv_data_transfer_over_time: keys are names of planners, values are vectors of total data transfered over time

            3. d_average_rate: keys are names of planners, values are the average rate
    """

    def get_m_rates_vs_time_single_planner(ind_planner):
        """
        Returns a matrix of size num_mc x num_time_samples of ue_rate_vs_time for the ind_planner-th planner.        
        """

        def complete_vecs(v_in, target_len):
            """
            Returns:                
                `v_out`: v_out[n] equals v_in[n] for n=0,..., argmax(v_in). The
                rest of entries up to target_len equal max(v_in)
            """
            ind_max = np.argmax(v_in)
            v_out = v_in[:ind_max]
            v_out = np.concatenate(
                (v_out, np.tile(v_in[ind_max], (target_len - len(v_out)))),
                axis=0)
            return v_out

        num_mc = len(llv_ue_rates_vs_time)
        # list with all mc realizations for the ind_planner-th planner
        lv_ue_rates_vs_time_this_planner = []
        for lv_ue_rates_vs_time in llv_ue_rates_vs_time:
            lv_ue_rates_vs_time_this_planner.append(
                lv_ue_rates_vs_time[ind_planner])

        # determine max length of the vectors in lv_ue_rates_vs_time
        num_time_samples = 0
        for ind_mc in range(num_mc):
            num_time_samples = np.maximum(
                num_time_samples,
                len(lv_ue_rates_vs_time_this_planner[ind_mc]))

        # num_mc x num_time_samples
        m_rates = np.array([
            complete_vecs(lv_ue_rates_vs_time_this_planner[ind_mc],
                          num_time_samples) for ind_mc in range(num_mc)
            if len(lv_ue_rates_vs_time_this_planner[ind_mc]) != 0
            or np.sum(lv_ue_rates_vs_time_this_planner[ind_mc]) > 0
        ])

        return m_rates

    dv_ue_rates_vs_time = {
        str(l_planners[ind_pp]):
        np.mean(get_m_rates_vs_time_single_planner(ind_pp), axis=0)
        for ind_pp in range(len(l_planners))
    }
    if samp_int is None:
        return dv_ue_rates_vs_time
    else:
        lv_cumsum_data_over_time = []
        l_average_rate = []
        for ind_pp in range(len(l_planners)):
            m_rates = get_m_rates_vs_time_single_planner(ind_pp)
            lv_cumsum_data_over_time.append(
                np.mean(np.cumsum(m_rates * samp_int, axis=1), axis=0))
            l_average_rate.append(
                np.mean(
                    np.cumsum(m_rates * samp_int, axis=1)[:, -1] /
                    (m_rates.shape[1] * samp_int)))
        dv_cumsum_data_over_time = {
            str(l_planners[ind_pp]): lv_cumsum_data_over_time[ind_pp]
            for ind_pp in range(len(l_planners))
        }

        d_average_rate = {
            str(l_planners[ind_pp]): l_average_rate[ind_pp]
            for ind_pp in range(len(l_planners))
        }

        return dv_ue_rates_vs_time, dv_cumsum_data_over_time, d_average_rate


def create_new_realization(f_environment,
                           f_channel,
                           lf_planners,
                           loc_bs=None,
                           dist_bs_ue=None,
                           min_ue_rate=None,
                           samp_int=None,
                           ue_speed=None,
                           time_duration=None,
                           num_pts_ue_grid=None,
                           dist_change=50):
    """
    If dist_bs_ue is
        + None, it is randomly generated.
        + Else, dist_bs_ue += dist_change * (2 * np.random.rand() - 1).
    """

    def get_dist_bs_ue(env):
        # if dist_bs_ue is too large, it cannot find a user inside the area
        # length with the given dist_bs_ue
        len_x = env.area_len[0]
        len_y = env.area_len[1]
        min_dist = .1 * (len_x + len_y) / 2
        max_dist = .9 * np.sqrt(np.sum(np.square(env.area_len[:2])))
        dist_bs_ue = np.random.uniform(min_dist, max_dist)
        return dist_bs_ue

    environment = f_environment()
    channel = f_channel(environment)
    l_planners = [f_planner(environment, channel) for f_planner in lf_planners]
    if loc_bs is None:
        loc_bs = environment.random_pts_on_street(1)[0]

    if dist_bs_ue is None:
        dist_bs_ue = get_dist_bs_ue(environment)
    else:
        dist_bs_ue += dist_change * (2 * np.random.rand() - 1)

    lm_ue_path = gen_path_on_street(
        environment,
        channel=channel,
        min_ue_rate=min_ue_rate,
        samp_int=samp_int,
        ue_speed=ue_speed,
        time_duration=time_duration,
        num_pts_grid=num_pts_ue_grid,  # [25, 25, 1]
        dist_bs_ue=dist_bs_ue,
        loc_bs=loc_bs,
        debug=0)

    return environment, channel, l_planners, loc_bs, lm_ue_path


def sim_metrics_vs_min_ue_rate_2serve_ue(f_environment,
                                         f_channel,
                                         lf_planners,
                                         samp_int=None,
                                         ue_speed=None,
                                         max_uav_speed=None,
                                         num_pts_ue_grid=None,
                                         time_duration=None,
                                         min_uav_rate=None,
                                         min_ue_rate=None,
                                         ind_mc_iter_start=0,
                                         num_mc_iter=1,
                                         loc_bs=None,
                                         dist_bs_ue=None,
                                         save_at=None,
                                         b_run_new=True,
                                         b_return_failure_only=False):
    """
    Serve either static or moving UE.
    
    """

    llv_ue_rates_vs_time = []
    for ind_mc in range(ind_mc_iter_start, ind_mc_iter_start + num_mc_iter):
        print(f'\n-- MC iteration: {ind_mc}')

        if b_run_new:
            environment, channel, l_planners, loc_bs, lm_ue_path = create_new_realization(
                f_environment,
                f_channel,
                lf_planners,
                loc_bs=loc_bs,
                dist_bs_ue=dist_bs_ue,
                min_ue_rate=min_ue_rate,
                samp_int=samp_int,
                ue_speed=ue_speed,
                time_duration=time_duration,
                num_pts_ue_grid=num_pts_ue_grid)

            if save_at is not None:
                save_realization(ind_mc,
                                 environment,
                                 channel,
                                 loc_bs,
                                 lm_ue_path,
                                 path_save_to=save_at)
        else:
            # load saved realization
            path_load_from = save_at + f"mc{ind_mc:03}.pickle"
            with open(path_load_from, 'rb') as file:
                d_realization = pickle.load(file)
            environment = d_realization["environment"]
            channel = d_realization["channel"]
            l_planners = [
                f_planner(environment, channel) for f_planner in lf_planners
            ]
            lm_ue_path = d_realization["lm_ue_path"]

        # num_planners x num_time_samples
        # b_run_new = True
        lv_ue_rates_vs_time = single_mc_rate(l_planners=l_planners,
                                             channel=channel,
                                             min_uav_rate=min_uav_rate,
                                             max_uav_speed=max_uav_speed,
                                             samp_int=samp_int,
                                             loc_bs=loc_bs,
                                             lm_ue_path=lm_ue_path,
                                             ind_mc=ind_mc,
                                             path_save_to=save_at,
                                             num_samples=len(lm_ue_path),
                                             b_run_new=b_run_new)

        # num_mc x num_planners x num_time_samples
        llv_ue_rates_vs_time.append(lv_ue_rates_vs_time)

    d_time_to_min_ue_rate, d_prob_failure, d_prob_outage = get_mean_time_to_min_ue_rate_n_prob(
        samp_int,
        l_planners,
        llv_ue_rates_vs_time=llv_ue_rates_vs_time,
        min_ue_rate=min_ue_rate,
        b_return_failure_only=b_return_failure_only)

    dv_ue_rates_vs_time, dv_cumsum_data_over_time, d_average_rate = get_ue_rates_vs_time(
        l_planners, llv_ue_rates_vs_time, samp_int=samp_int)

    return d_time_to_min_ue_rate, d_prob_failure, d_prob_outage, dv_ue_rates_vs_time, dv_cumsum_data_over_time, d_average_rate


def sim_optimality_vs_dist_bs_static_ue(f_environment,
                                        f_channel,
                                        lf_planners,
                                        percent_pts_disable=None,
                                        samp_int=None,
                                        max_uav_speed=None,
                                        ind_mc_iter_start=0,
                                        num_mc_iter=1,
                                        dist_bs_ue=500,
                                        min_ue_rate=None,
                                        loc_bs=None,
                                        loc_ue=None,
                                        save_at=None,
                                        b_run_new=True):

    ll_is_optimal = []

    for ind_mc in range(ind_mc_iter_start, ind_mc_iter_start + num_mc_iter):
        print(f'\n-- MC iteration: {ind_mc}')

        if b_run_new:
            assert percent_pts_disable is not None and percent_pts_disable < 1, 'percentage_amount_pts_to_disable must be in (0, 1)'

            environment = f_environment()
            environment.fly_grid.disable_at_random(
                percent_pts_disable=percent_pts_disable)

            channel = f_channel(environment)

            l_planners = [
                f_planner(environment, channel) for f_planner in lf_planners
            ]

            if loc_ue is None:
                assert dist_bs_ue is not None and loc_bs is not None, 'dist_bs_ue and loc_bs cannot be None when loc_ue is None'
                dist_bs_ue += 100 * (2 * np.random.rand() - 1)
                loc_ue = get_random_ue_loc(environment,
                                           channel=channel,
                                           min_ue_rate=min_ue_rate,
                                           dist_bs_ue=dist_bs_ue,
                                           loc_bs=loc_bs)

            if save_at is not None:
                save_realization(ind_mc,
                                 environment,
                                 channel,
                                 loc_bs,
                                 loc_ue,
                                 path_save_to=save_at)
        else:
            # load saved realization
            path_load_from = save_at + f"mc{ind_mc:03}.pickle"
            with open(path_load_from, 'rb') as file:
                d_realization = pickle.load(file)
            environment = d_realization["environment"]
            channel = d_realization["channel"]
            l_planners = [
                f_planner(environment, channel) for f_planner in lf_planners
            ]
            loc_ue = d_realization["lm_ue_path"]

        def single_mc_optimality(l_planners=l_planners,
                                 max_uav_speed=max_uav_speed,
                                 samp_int=samp_int,
                                 loc_bs=loc_bs,
                                 loc_ue=loc_ue,
                                 save_at=None,
                                 b_run_new=True):

            def save_or_load_status(ind_mc,
                                    planner_name,
                                    status,
                                    save_at,
                                    b_save=True):
                save_at = save_at + planner_name
                if b_save:
                    os.makedirs(save_at, exist_ok=True)
                    save_at += f"/mc{ind_mc:03}_output.pickle"

                    d_results = {
                        "status": status,
                    }

                    # save to pickle file
                    with open(save_at, 'wb') as file:
                        pickle.dump(d_results, file)
                else:
                    # load results
                    path_load_from = save_at + f"/mc{ind_mc:03}_output.pickle"
                    with open(path_load_from, 'rb') as file:
                        d_results = pickle.load(file)

                    return d_results["status"]

            l_is_optimal = []

            for planner in l_planners:
                if b_run_new:
                    lm_path = planner.plan_path_to_serve_static_ue(
                        bs_loc=loc_bs,
                        ue_loc=loc_ue,
                        samp_int=samp_int,  #s
                        max_uav_speed=max_uav_speed)

                    # environment.l_users = [loc_bs, loc_ue]
                    # environment.plot()

                    if lm_path is not None:
                        status = planner.b_no_waiting_or_lifting

                    else:
                        print(f'--- No path found for {planner.name}')
                        status = np.nan

                    # save results
                    save_or_load_status(ind_mc,
                                        planner.name,
                                        status,
                                        save_at=save_at,
                                        b_save=True)
                else:
                    assert save_at is not None
                    status = save_or_load_status(ind_mc,
                                                 planner.name,
                                                 status=None,
                                                 save_at=save_at,
                                                 b_save=False)

                l_is_optimal.append(status)

            return l_is_optimal

        l_is_optimal = single_mc_optimality(l_planners=l_planners,
                                            max_uav_speed=max_uav_speed,
                                            samp_int=samp_int,
                                            loc_bs=loc_bs,
                                            loc_ue=loc_ue,
                                            save_at=save_at,
                                            b_run_new=b_run_new)

        # num_mc x num_planners
        if l_is_optimal is not None:
            ll_is_optimal.append(l_is_optimal)

    return ll_is_optimal
