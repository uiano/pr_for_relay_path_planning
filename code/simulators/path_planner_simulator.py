import numpy as np
from relays.path_planners import PathPlanner


def is_out_pad_buildings(env, padding, pt):
    # Padding around buildings to avoid the noncoincide between the buildings and slf grid
    v_safety = padding * np.array([[-1, 1]])
    block_limits_x_w_padding = env.block_limits_x + v_safety
    block_limits_y_w_padding = env.block_limits_y + v_safety

    if (np.sum(block_limits_x_w_padding - pt[0] >= 0) % 2
            == 1) and (np.sum(block_limits_y_w_padding - pt[1] >= 0) % 2 == 1):
        return False
    else:
        return True


def random_ue_bs_w_padding_building_wo_los(env, padding=5, num_pts=2):

    l_pts = []

    while len(l_pts) < num_pts:
        pt_random = env.random_pts_on_street(1)[0]
        if is_out_pad_buildings(env, padding, pt_random):
            l_pts.append(pt_random)

    if env.slf.line_integral(l_pts[0], l_pts[1], mode="c")[0] == 0:
        return random_ue_bs_w_padding_building_wo_los(env,
                                                      padding=padding,
                                                      num_pts=num_pts)
    else:
        return l_pts


def random_ue_bs_w_padding_building_lower_than_rate(env,
                                                    channel,
                                                    padding=5,
                                                    max_direct_rate=None):
    """
    This function returns two locations such that the rate between them is never
    greater than `max_direct_rate`.
    
    """

    assert max_direct_rate is not None

    l_pts = []

    while len(l_pts) < 2:
        pt_random = env.random_pts_on_street(1)[0]
        if is_out_pad_buildings(env, padding, pt_random):
            l_pts.append(pt_random)

    bs_loc = l_pts[0]
    ue_loc = l_pts[1]

    if channel.dbgain_to_capacity(channel.dbgain(bs_loc,
                                                 ue_loc)) >= max_direct_rate:
        return random_ue_bs_w_padding_building_lower_than_rate(
            env, channel, padding=padding, max_direct_rate=max_direct_rate)
    else:
        return l_pts


def random_ue_bs_w_padding_building_wo_los_w_dist(env,
                                                  padding=5,
                                                  bs_ue_dist=10):
    def random_pts_w_dist(bs_loc):
        theta = np.random.uniform(0, 2 * np.pi)

        ue_x = bs_loc[0] + bs_ue_dist * np.cos(theta)
        ue_y = bs_loc[1] + bs_ue_dist * np.sin(theta)

        return np.array([ue_x, ue_y, 0])

    def is_in_area_len(pt):

        is_in_x_limits = (0 <= pt[0]) and (pt[0] <= env.area_len[0] - padding)
        is_in_y_limits = (0 <= pt[1]) and (pt[1] <= env.area_len[0] - padding)

        if is_in_x_limits and is_in_y_limits:
            return True
        else:
            return False

    # random bs
    is_in_building = True
    while is_in_building:
        bs_loc = env.random_pts_on_street(1)[0]
        if is_out_pad_buildings(env, padding, bs_loc):
            is_in_building = False

    # random ue with distance with bs
    is_in_building = True
    while is_in_building:
        ue_loc = random_pts_w_dist(bs_loc)
        if is_out_pad_buildings(env, padding,
                                ue_loc) and is_in_area_len(ue_loc):
            is_in_building = False

    if env.slf.line_integral(bs_loc, ue_loc, mode="c")[0] == 0:
        return random_ue_bs_w_padding_building_wo_los_w_dist(
            env, padding=padding, bs_ue_dist=bs_ue_dist)

    else:
        return [bs_loc, ue_loc]


def single_mc(d_mean_time_n_rate, l_planners, bs_loc, ue_loc, samp_int,
              max_uav_speed):
    """
        Run single MC iteration,
        If no path found, return None
        Otherwise, update d_mean_time
    """

    for planner in l_planners:

        lm_path = planner.plan_path(
            bs_loc=bs_loc,
            ue_loc=ue_loc,
            samp_int=samp_int,  #s
            max_uav_speed=max_uav_speed)

        assert lm_path is not None

        time_n_rate = planner.time_to_connect(lm_path, samp_int, ue_loc)

        assert time_n_rate is not None

        print(
            f"----- {planner.name}, time: {time_n_rate[0]}, rate: {time_n_rate[1]},"
        )

        d_mean_time_n_rate[planner.name].append(time_n_rate)

    return d_mean_time_n_rate


def mean_time_to_connect(environment,
                         l_planners,
                         samp_int=1 / 5,
                         max_uav_speed=10,
                         num_mc_iter=1,
                         padding=5,
                         num_pts=2):
    """Returns a dict whose keys are the names of the path planners in
    `l_planners` and the values are the MC estimates of the mean time to connect."""

    d_mean_time_n_rate = {planner.name: [] for planner in l_planners}

    for ind_mc in range(num_mc_iter):

        l_pts = random_ue_bs_w_padding_building_wo_los(environment,
                                                       padding=padding,
                                                       num_pts=num_pts)
        bs_loc = l_pts[0]
        ue_loc = l_pts[1]

        print(f"MC: {ind_mc}")

        d_mean_time_n_rate = single_mc(d_mean_time_n_rate, l_planners, bs_loc,
                                       ue_loc, samp_int, max_uav_speed)

    return {key: np.mean(val) for key, val in d_mean_time_n_rate.items()}


def mean_time_vs_distance(environment,
                          l_planners,
                          samp_int=1 / 5,
                          max_uav_speed=10,
                          padding=5,
                          bs_ue_dist=None,
                          num_mc_iter=1):
    """Returns a dict whose keys are the names of the path planners in
    `l_planners` and the values are the MC estimates of the mean time to connect."""

    assert bs_ue_dist is not None

    d_mean_time_n_rate = {planner.name: [] for planner in l_planners}

    for ind_mc in range(num_mc_iter):

        l_pts = random_ue_bs_w_padding_building_wo_los_w_dist(
            environment, padding=padding, bs_ue_dist=bs_ue_dist)

        bs_loc = l_pts[0]
        ue_loc = l_pts[1]

        print(f"MC: {ind_mc}, bs_loc: {bs_loc}, ue_loc: {ue_loc}")

        d_mean_time_n_rate = single_mc(d_mean_time_n_rate, l_planners, bs_loc,
                                       ue_loc, samp_int, max_uav_speed)

    return {
        key: np.mean(val, axis=0)
        for key, val in d_mean_time_n_rate.items()
    }


def single_mc_time_n_rate(l_planners, channel, min_uav_rate, min_ue_rate,
                          samp_int, max_uav_speed, bs_loc, ue_loc):
    """
            Returns: 

            `l_time_to_min_ue_rate`: list whose n-th entry is the time needed by
            l_planners[n] to attain `min_ue_rate` if it attains this rate. Else it
            is None. 

            `l_ue_rates_vs_time`: list whose n-th entry is the rate vs. time of l_planners[n].

        """
    def get_time_to_min_ue_rate(v_ue_rate):
        for ind_rate in range(len(v_ue_rate)):
            if v_ue_rate[ind_rate] >= min_ue_rate:
                return ind_rate * samp_int
        return None

    l_time_to_min_ue_rate = []
    lv_ue_rates_vs_time = []
    for planner in l_planners:

        lm_path = planner.plan_path(
            bs_loc=bs_loc,
            ue_loc=ue_loc,
            samp_int=samp_int,  #s
            max_uav_speed=max_uav_speed)

        if lm_path is not None:
            _, v_ue_rate = PathPlanner.rate_from_path(channel,
                                                      lm_path,
                                                      min_uav_rate,
                                                      bs_loc,
                                                      ue_loc=ue_loc)

        else:
            v_ue_rate = np.array([])

        # Compute time to connect
        time_to_min_ue_rate = get_time_to_min_ue_rate(v_ue_rate)
        lv_ue_rates_vs_time.append(v_ue_rate)
        l_time_to_min_ue_rate.append(time_to_min_ue_rate)

    return l_time_to_min_ue_rate, lv_ue_rates_vs_time


def get_time_n_prob(l_planners, ll_times_to_min_ue_rate):
    """
            Args:
                + ll_times_to_min_ue_rate: a list of length num_mc_iter. Each entry is a list of length number_planners consiting of time to connect of each planner.

            Returns:
                + d_time_to_min_ue_rate: keys are name of planners, values are mean time to obtain min_ue_rate
                + d_prob_of_failure: keys are name of planners

        """

    # num_path_planners x num_mc
    m_times_to_min_ue_rate = np.array(ll_times_to_min_ue_rate).T

    m_failures = (m_times_to_min_ue_rate == None)

    m_times_to_min_ue_rate[m_failures] = 0

    # (num_planners,)
    v_num_successes = np.sum(np.logical_not(m_failures), axis=1)

    v_mean_time_to_min_ue_rate = np.sum(
        m_times_to_min_ue_rate, axis=1) / np.maximum(1, v_num_successes)

    v_mean_time_to_min_ue_rate[v_num_successes == 0] = np.nan

    v_prob_of_failure = np.mean(m_failures, axis=1)

    d_time_to_min_ue_rate = {
        str(l_planners[ind_pp]): v_mean_time_to_min_ue_rate[ind_pp]
        for ind_pp in range(len(l_planners))
    }

    d_prob_of_failure = {
        str(l_planners[ind_pp]): v_prob_of_failure[ind_pp]
        for ind_pp in range(len(l_planners))
    }
    return d_time_to_min_ue_rate, d_prob_of_failure


def get_ue_rates_vs_time(llv_ue_rates_vs_time, ind_planner):
    """
        Args:

        `llv_ue_rates_vs_time`: list of length num_mc of lists of length num_planners of vectors. 

        Returns a vector with the ue_rate_vs_time for the ind_planner-th planner.
        
        """
    def complete_vecs(v_in, target_len):
        """Returns:
            
            `v_out`: v_out[n] equals v_in[n] for n=0,..., argmax(v_in). The rest
            of entries up to target_len equal max(v_in)
            """
        ind_max = np.argmax(v_in)
        v_out = v_in[:ind_max]
        v_out = np.concatenate(
            (v_out, np.tile(v_in[ind_max], (target_len - len(v_out)))), axis=0)
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
            num_time_samples, len(lv_ue_rates_vs_time_this_planner[ind_mc]))

    # num_mc x num_time_samples
    m_rates = np.array([
        complete_vecs(lv_ue_rates_vs_time_this_planner[ind_mc],
                      num_time_samples) for ind_mc in range(num_mc)
        if len(lv_ue_rates_vs_time_this_planner[ind_mc]) != 0
    ])

    return np.mean(m_rates, axis=0)


def sim_min_ue_rate_metrics(environment,
                            channel,
                            l_planners,
                            samp_int,
                            max_uav_speed,
                            padding,
                            min_uav_rate,
                            min_ue_rate,
                            num_mc_iter=1,
                            bs_ue_dist=None):
    """
    Args:

    `bs_ue_dist`: if not None, then the UE and BS locations are generated such
    that their distance is `bs_ue_dist`. If None, then these positions are
    generated with a random distance but it is imposed that their direct rate is
    not greater than `min_ue_rate`.


    Returns three dicts whose keys are the names of the path planners in
    `l_planners` and the values are:

    - 'd_time_to_min_ue_rate':  MC average of the min time to reach a rate of
      min_ue_rate. Those MC iterations where this min rate is never achieved are
      not considered. 

    - 'd_prob_of_failure': fraction of MC realizations where this min rate was
       eventually achieved. 
    
    - 'd_ue_rates_vs_time`
    
    """
    def get_ue_bs_locs():
        if bs_ue_dist is None:
            bs_loc, ue_loc = random_ue_bs_w_padding_building_lower_than_rate(
                environment,
                channel,
                padding=padding,
                max_direct_rate=min_ue_rate)
        else:
            bs_loc, ue_loc = random_ue_bs_w_padding_building_wo_los_w_dist(
                environment, padding=padding, bs_ue_dist=bs_ue_dist)

        return bs_loc, ue_loc

    ll_times_to_min_ue_rate = []
    llv_ue_rates_vs_time = []
    while True:

        print(f'----- mc iteration: {len(ll_times_to_min_ue_rate)}')

        bs_loc, ue_loc = get_ue_bs_locs()

        l_time_to_min_ue_rate, lv_ue_rates_vs_time = single_mc_time_n_rate(
            l_planners,
            channel=channel,
            min_uav_rate=min_uav_rate,
            samp_int=samp_int,
            max_uav_speed=max_uav_speed,
            min_ue_rate=min_ue_rate,
            bs_loc=bs_loc,
            ue_loc=ue_loc)

        ll_times_to_min_ue_rate.append(l_time_to_min_ue_rate)

        llv_ue_rates_vs_time.append(lv_ue_rates_vs_time)

        if len(ll_times_to_min_ue_rate) >= num_mc_iter:
            break

    d_time_to_min_ue_rate, d_prob_of_failure = get_time_n_prob(
        l_planners, ll_times_to_min_ue_rate)

    d_ue_rates_vs_time = {
        str(l_planners[ind_pp]): get_ue_rates_vs_time(llv_ue_rates_vs_time,
                                                      ind_pp)
        for ind_pp in range(len(l_planners))
    }

    return d_time_to_min_ue_rate, d_prob_of_failure, d_ue_rates_vs_time
