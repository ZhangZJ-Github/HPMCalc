# -*- coding: utf-8 -*-
# @Time    : 2025/1/11 21:16
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : side_arm_length_calculator.py
# @Software: PyCharm

import matplotlib

matplotlib.use('tkagg')

from scipy.optimize import minimize

from theory.rfCompressor.crosslike._base import *

plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_target = 9.3e9
f_target_GHz = f_target / 1e9


def __get_output_signal_of_the_network_connects_with_delay_shorts(  # s,
        nw_cross: skrf.Network,
        port_ids_to_delay_shorts, beta_l_delay_shorts, index_of_port_to_input_signal: int = 0,
        a_input_port=1 + 0j,
        f_target=f_target):
    z = 1j * numpy.tan(beta_l_delay_shorts)
    return calculate_signals_outgoing(
        nw_cross.interpolate([f_target]).s[0],
        # s,
        list(port_ids_to_delay_shorts),
        (z - 1) / (z + 1),
        index_of_port_to_input_signal, a_input_port).A.ravel()


def __get_beta_l_to_min_power_output(
        # nw_cross: skrf.Network,
        crosslike: CrossLikeNetwork,
        # port_ids_to_delay_shorts,# beta_l_delay_shorts,
        # port_ids_out,
        beta_l_thru,
        beta_l_SC_range: numpy.ndarray,
        index_of_port_to_input_signal: int = 0,
        # a_input_port=1 + 0j,
        f_target=f_target, ):
    port_id_thru = port_ids_thru_arm  # [1]
    # port_ids_to_SC =port_ids_to_SC#[4, 5]
    # port_ids_mismatched = port_id_thru + port_ids_to_SC
    # get_power_out_ratio = lambda beta_l_delay_shorts: numpy.sum(
    #     numpy.abs(__get_output_signal_of_the_network_connects_with_delay_shorts(
    #         nw_cross, port_ids_mismatched, [beta_l_delay_shorts[0], beta_l_delay_shorts[1], beta_l_delay_shorts[1]],
    #         index_of_port_to_input_signal,
    #         1 + 0j, f_target)[port_ids_out]) ** 2, axis=-1)
    __get_output_signals = lambda beta_l_delay_shorts: __get_output_signal_of_the_network_connects_with_delay_shorts(
        crosslike.nw_cross,
        port_ids_mismatched, [beta_l_thru, beta_l_delay_shorts[0], #beta_l_delay_shorts[0],
                              ],
        index_of_port_to_input_signal, 1 + 0j, f_target
    )
    get_power_out_ratio = lambda beta_l_delay_shorts: numpy.sum(
        numpy.abs(__get_output_signals(beta_l_delay_shorts)[port_ids_out]) ** 2, axis=-1)
    # get_power_out_ratio = lambda beta_l_delay_shorts: crosslike.get_total_out_power_ratio(
    #     # nw_cross,
    #     dict(zip(port_ids_mismatched,
    #              [beta_l_thru, beta_l_delay_shorts[0], ])),
    #     port_ids_out,
    #     index_of_port_to_input_signal, f_target
    # )
    # beta_l_range_for_finding_min = numpy.array([
    #     [0.2, 0.7],
    #     [0.9, 1.1]
    # ]) * numpy.pi
    beta_l_range_for_finding_min = beta_l_SC_range
    min_res = minimize(get_power_out_ratio,
                       x0=(beta_l_range_for_finding_min[:, 0] + beta_l_range_for_finding_min[:, 1]) / 2,
                       bounds=beta_l_range_for_finding_min,
                       # tol = 0.01/19
                       )
    logger.info("最优值在参数空间的相对位置：(%s)" % ((min_res.x - beta_l_range_for_finding_min[:, 0]) / (
            beta_l_range_for_finding_min[:, 1] - beta_l_range_for_finding_min[:, 0])))

    logger.info(min_res)
    ratio_power_outgoing_side_arm = numpy.max(numpy.abs(__get_output_signals(min_res.x)[port_ids_to_SC]) ** 2, axis=-1)
    # ratio_power_outgoing_side_arm = max(
    #     [numpy.max(numpy.abs(__get_output_signal_of_the_network_connects_with_delay_shorts(
    #         nw_cross, port_id_thru + [port_id_to_SC], [beta_l_thru, min_res.x[0], ],
    #         index_of_port_to_input_signal,
    #                   1 + 0j, f_target
    #     ) ** 2)[port_ids_to_SC[(i+1)%len(port_ids_to_SC)]]) for i,port_id_to_SC in enumerate(port_ids_to_SC)])

    logger.info(
        "Power passing forward one of the side arms = %.2f%%" % (ratio_power_outgoing_side_arm * 100))
    P0 = 1e9
    P = P0 * ratio_power_outgoing_side_arm

    max_E_inside_switch_cav_when_no_out = 4 * (numpy.abs(z0_WG_side_arm_interpolator(f_target_GHz)) * P / (
            rect_WG_to_side_arms.a * rect_WG_to_side_arms.b)) ** 0.5

    max_E_inside_ESWG = 4 * (numpy.abs(z0_WG_thru_arm_interpolator(f_target_GHz)) * P0 / (
            rect_WG_to_thru_arm.a * rect_WG_to_thru_arm.b)) ** 0.5
    max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG = max_E_inside_switch_cav_when_no_out / max_E_inside_ESWG
    logger.info(
        "When the power output is zero, maximum E field inside the side arm is %.2e V/m, which is %.2f%% of that inside the ESWG (%.2e V/m)\n(assuming the power input from port 1 is 1 GW)" % (
            max_E_inside_switch_cav_when_no_out,
            max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG * 100, max_E_inside_ESWG,))

    return min_res, max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG


def __get_beta_l_to_min_or_max_power_output(cross_like: CrossLikeNetwork,
                                            # nw_cross: skrf.Network,
                                            # port_ids_to_delay_shorts,# beta_l_delay_shorts,
                                            port_ids_out,
                                            # beta_l_thru,
                                            beta_l_range_for_finding_min, beta_l_range_for_finding_max,
                                            # beta_l_range: numpy.ndarray,
                                            index_of_port_to_input_signal: int = 0,
                                            # a_input_port=1 + 0j,
                                            f_target=f_target, ):
    """

    :param nw_cross:
    :param port_ids_to_delay_shorts:
    :param beta_l_delay_shorts:
    :param port_ids_out:
    :param beta_l_range: 形如[[min,max],[min,max],...]
    :param index_of_port_to_input_signal:
    :param a_input_port:
    :param f_target:
    :return:
    """

    port_id_thru = port_ids_thru_arm
    __get_output_signals = lambda beta_l_delay_shorts: __get_output_signal_of_the_network_connects_with_delay_shorts(
        cross_like.nw_cross,
        port_ids_mismatched, [beta_l_delay_shorts[0], beta_l_delay_shorts[1],# beta_l_delay_shorts[1],
                              ],
        index_of_port_to_input_signal, 1 + 0j, f_target
    )
    get_power_out_ratio = lambda beta_l_delay_shorts: numpy.sum(
        numpy.abs(__get_output_signals(beta_l_delay_shorts)[port_ids_out]) ** 2, axis=-1)

    min_res = minimize(get_power_out_ratio,
                       x0=(beta_l_range_for_finding_min[:, 0] + beta_l_range_for_finding_min[:, 1]) / 2,
                       bounds=beta_l_range_for_finding_min,
                       # tol = 0.01/19
                       )
    logger.info("最优值在参数空间的相对位置：(%s)" % ((min_res.x - beta_l_range_for_finding_min[:, 0]) / (
            beta_l_range_for_finding_min[:, 1] - beta_l_range_for_finding_min[:, 0])))

    logger.info(min_res)
    ratio_power_outgoing_side_arm = numpy.max(numpy.abs(__get_output_signals(min_res.x)[port_ids_to_SC]) ** 2, axis=-1)

    logger.info(
        "Power passing forward one of the side arms = %.2f%%" % (ratio_power_outgoing_side_arm * 100))
    P0 = 1e9
    P = P0 * ratio_power_outgoing_side_arm

    max_E_inside_switch_cav_when_no_out = 4 * (numpy.abs(z0_WG_side_arm_interpolator(f_target_GHz)) * P / (
            rect_WG_to_side_arms.a * rect_WG_to_side_arms.b)) ** 0.5

    max_E_inside_ESWG = 4 * (numpy.abs(z0_WG_thru_arm_interpolator(f_target_GHz)) * P0 / (
            rect_WG_to_thru_arm.a * rect_WG_to_thru_arm.b)) ** 0.5
    max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG = max_E_inside_switch_cav_when_no_out / max_E_inside_ESWG
    logger.info(
        "When the power output is zero, maximum E field inside the side arm is %.2e V/m, which is %.2f%% of that inside the ESWG (%.2e V/m)\n(assuming the power input from port 1 is 1 GW)" % (
            max_E_inside_switch_cav_when_no_out,
            max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG * 100, max_E_inside_ESWG,))

    max_res = minimize(lambda beta_l_SC: - get_power_out_ratio([min_res.x[0], beta_l_SC[0]]),
                       x0=(beta_l_range_for_finding_max[:, 0] + beta_l_range_for_finding_max[:, 1]) / 2,
                       bounds=beta_l_range_for_finding_max,  # tol = 0.01/19
                       )

    logger.info(max_res)
    logger.info("最优值在参数空间的相对位置：(%s)" % ((max_res.x - beta_l_range_for_finding_max[:, 0]) / (
            beta_l_range_for_finding_max[:, 1] - beta_l_range_for_finding_max[:, 0])))
    # nw_maximize_power_out = cross_like.get_nw_connected_with_delay_shorts(
    #     # nw_cross,
    #     dict(zip(port_ids_mismatched, [min_res.x[0], max_res.x[0], max_res.x[0]], )))[-1]
    l_thru = min_res.x[0] / beta_thru_arm_interpolator(f_target_GHz)
    l_SC_min_out = min_res.x[1] / beta_side_arm_interpolator(f_target_GHz)
    l_SC_max_out = max_res.x[0] / beta_side_arm_interpolator(f_target_GHz)

    vectf_calculate_signals_outgoing = numpy.vectorize(
        lambda beta_l_delay_shorts, f_target:
        __get_output_signal_of_the_network_connects_with_delay_shorts(
            cross_like.nw_cross, port_ids_mismatched, beta_l_delay_shorts,
            index_of_port_to_input_signal=index_of_port_to_input_signal,
            a_input_port=1 + 0j,
            f_target=f_target),
        signature="(Np),()->(N_allports)")

    if __name__ == '__main__':
        f = skrf_frequency.f
        # nports_of_the_final_nw = 3  # nw_minimize_power_out.nports
        # plt.ion()
        fig, axs = plt.subplots(2, 1, figsize=(4, 3), constrained_layout=True, sharex=True)

        _S_square = 1 - numpy.abs(vectf_calculate_signals_outgoing(
            numpy.array((beta_thru_arm_interpolator(f_GHz) * l_thru, beta_side_arm_interpolator(f_GHz) * l_SC_min_out,
                         # beta_side_arm_interpolator(f_GHz) * l_SC_min_out,
                         )).T,
            f
        )[:, 0]) ** 2
        axs[0].plot(f / 1e9, (_S_square), )
        axs[1].plot(f / 1e9, 10 * numpy.log10(_S_square), )

        # nw_minimize_power_out.plot_s_db()

        plt.figure()
        plt.plot(f / 1e9, 1 - numpy.abs(vectf_calculate_signals_outgoing(
            numpy.array((beta_thru_arm_interpolator(f_GHz) * l_thru, beta_side_arm_interpolator(f_GHz) * l_SC_max_out,
                         # beta_side_arm_interpolator(f_GHz) * l_SC_max_out,
                         )).T,
            f
        )[:, 0])**2, )
        plt.ylim(0, 1)


        # plt.ioff()

    return (min_res, max_res,
            # nw_minimize_power_out, nw_maximize_power_out,
            max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG)

if __name__ == '__main__':
    s_interpolator_on_dataset, z0_interpolator_on_dataset, data_set = get_interpolator_on_dataset(
        r"E:\CSTprojects\rfCompressor\cascadedHT\why_2hole\HplaneCrossLike.cst", [  # "SCcenterDz",
            'OutDz1', ])
    s_interpolator_on_dataset: LinearNDInterpolator
    example_data = data_set[list(data_set.keys())[0]]
    f_GHz = example_data['f']
    skrf_frequency = skrf.Frequency.from_f(f_GHz, unit='GHz')
    z0 = example_data['z0']
    # N_considered_modes = 3
    rect_WG_to_thru_arm = RectangularWaveguide(frequency=skrf_frequency,
                                               # z0_port = nw.z0[:,__name_map_from_CST_to_skrf[2]],
                                               a=30e-3, b=15e-3, mode_type='te', m=1, n=0)

    # 开关腔的雏形波导
    rect_WG_to_side_arms = RectangularWaveguide(frequency=skrf_frequency,
                                                a=30e-3, b=15e-3, mode_type='te', m=1, n=0)
    # rect_WGs = [rect_WG_to_thru_arm, rect_WG_to_side_arms, rect_WG_to_side_arms]
    rect_WG_to_outlet = RectangularWaveguide(frequency=skrf_frequency,
                                             # z0_port = nw.z0[:,__name_map_from_CST_to_skrf[2]],
                                             a=18e-3, b=15e-3, mode_type='te', m=1, n=0)
    z0_WG_side_arm_interpolator = interp1d(f_GHz, rect_WG_to_side_arms.z0)
    z0_WG_thru_arm_interpolator = interp1d(f_GHz, rect_WG_to_thru_arm.z0)
    beta_thru_arm_interpolator = interp1d(f_GHz, rect_WG_to_thru_arm.beta)
    beta_side_arm_interpolator = interp1d(f_GHz, rect_WG_to_side_arms.beta)
    beta_l_thru, beta_l_SC = numpy.linspace(0, 2 * numpy.pi, 50), numpy.linspace(0, 2 * numpy.pi, 51)
    beta_l_thru_for_finding_min = numpy.linspace(0, 1 * numpy.pi, 30)

    BETA_L_THRU, BETA_L_SC = numpy.meshgrid(beta_l_thru, beta_l_SC, indexing='ij')
    port_ids_thru_arm = [1, ]
    port_ids_to_SC = [3]
    port_ids_mismatched = port_ids_thru_arm + port_ids_to_SC
    port_ids_out = [2, ]
    port_ids_out_in_the_final_network = CrossLikeNetwork.get_port_ids_in_new_nw(port_ids_out, port_ids_mismatched)
    map_from_port_id_to_matched_waveguide = dict(
        zip(port_ids_mismatched, [rect_WG_to_thru_arm, rect_WG_to_side_arms, ]))
    plt.ioff()
    for i, (OutDz1) in enumerate(s_interpolator_on_dataset.x):
        str_indexing_parameter = "OutDz1 = %.0f mm" % (OutDz1)
        logger.info(str_indexing_parameter)
        nw_cross = skrf.Network(frequency=skrf_frequency,
                                s=s_interpolator_on_dataset((OutDz1)), z0=z0
                                )
        cross_like = CrossLikeNetwork(nw_cross, map_from_port_id_to_matched_waveguide)
        vectf__get_output_signal_of_the_network_connects_with_delay_shorts = numpy.vectorize(
            lambda beta_l_thru, beta_l_SC1,: __get_output_signal_of_the_network_connects_with_delay_shorts(
                nw_cross,
                port_ids_mismatched,
                [beta_l_thru,
                 beta_l_SC1, ]),
            signature="(),()->(m)"
        )


        def f1(beta_l_thru):
            ret = __get_beta_l_to_min_power_output(cross_like,  # port_ids_out,
                                                   beta_l_thru,
                                                   numpy.array([[0, 1.5 * numpy.pi]]), )
            return numpy.array([ret[0].x[0], ret[1]])


        vectf__get_beta_l_to_min_power_output = numpy.vectorize(lambda beta_l_thru: f1(beta_l_thru),
                                                                signature="()->(2)")

        t1 = time.time()
        S_allports_0 = vectf__get_output_signal_of_the_network_connects_with_delay_shorts(BETA_L_THRU, BETA_L_SC,
                                                                                          )
        beta_l_SC_to_min_out_at_different_beta_l_thru, maxE_ratio = vectf__get_beta_l_to_min_power_output(
            beta_l_thru_for_finding_min).T
        logger.info("Time cost = %.10f s" % (time.time() - t1))
        # logger.info(numpy.abs(S_allports_0).max())

        fig, axs = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True, sharex=True)
        d_total_out = 0.05
        # nw3.plot_s_mag()
        fig: plt.Figure
        axs: typing.List[plt.Axes]

        cf = axs[0].contourf(BETA_L_THRU / numpy.pi, BETA_L_SC / numpy.pi,
                             numpy.sum(numpy.abs(S_allports_0[:, :, port_ids_out]) ** 2, axis=-1),
                             levels=numpy.arange(0, 1.0 + d_total_out, d_total_out)
                             )
        c_ = axs[0].contour(BETA_L_THRU / numpy.pi, BETA_L_SC / numpy.pi,
                            numpy.sum(numpy.abs(S_allports_0[:, :, port_ids_out]) ** 2, axis=-1),
                            [0.5]
                            )
        plt.clabel(c_, inline=True,  # fontsize=10,
                   )
        fixed_beta_l_SC_to_min_out_at_different_beta_l_thru = numpy.unwrap(
            (beta_l_SC_to_min_out_at_different_beta_l_thru) % numpy.pi, period=numpy.pi)
        fixed_beta_l_SC_to_min_out_at_different_beta_l_thru = fixed_beta_l_SC_to_min_out_at_different_beta_l_thru + (
                fixed_beta_l_SC_to_min_out_at_different_beta_l_thru.min() % numpy.pi - fixed_beta_l_SC_to_min_out_at_different_beta_l_thru.min())
        axs[0].scatter(beta_l_thru_for_finding_min / numpy.pi,
                       fixed_beta_l_SC_to_min_out_at_different_beta_l_thru / numpy.pi
                       , c='r')
        fig.colorbar(cf, ax=axs)
        fig.supxlabel(r"$(\beta l)_{thru} / \pi$")
        axs[0].set_ylabel(r"$(\beta l)_{SC} / \pi$")
        fig.suptitle(str_indexing_parameter)
        axs[1].plot(beta_l_thru_for_finding_min / numpy.pi, maxE_ratio ** 2)
        max_idx = numpy.argmax(maxE_ratio ** 2)
        max_pt = beta_l_thru_for_finding_min[max_idx] / numpy.pi, maxE_ratio[max_idx] ** 2
        axs[1].scatter(*max_pt, label="Max (%.2f, %.2f)" % (*max_pt,))
        axs[1].legend()
        # axs[1].set_ylim(0, 1)
        axs[1].set_ylabel(r"$\left(E_{max}^{SC}/E_{max}\right)^2$")
        plt.savefig(".out/%d_(%s).png" % (i, str_indexing_parameter), dpi=200)
        plt.close()

    plt.ion()
    res = __get_beta_l_to_min_or_max_power_output(
        CrossLikeNetwork(
            skrf.Network(frequency=skrf_frequency,
                         s=s_interpolator_on_dataset(( 1.0e-03)),
                         # z0=numpy.array((rect_WG_to_thru_arm.z0, rect_WG_to_thru_arm.z0,
                         #                 rect_WG_to_outlet.z0, rect_WG_to_outlet.z0,
                         #                 rect_WG_to_side_arms.z0, rect_WG_to_side_arms.z0
                         #                 )).reshape((-1, 6))
                         z0=z0
                         ), map_from_port_id_to_matched_waveguide),
        port_ids_out,
        numpy.array([[0.3, 0.35], [0.3, 0.7]]) * numpy.pi,
        numpy.array([[0., 0.3]]) * numpy.pi,
    )
