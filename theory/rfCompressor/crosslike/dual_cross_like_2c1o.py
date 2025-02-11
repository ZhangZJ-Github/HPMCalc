# -*- coding: utf-8 -*-
# @Time    : 2025/1/11 21:16
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : side_arm_length_calculator.py
# @Software: PyCharm
import cst.results
import cst.results
import matplotlib

matplotlib.use('tkagg')
import skrf
from skrf.media.rectangularWaveguide import RectangularWaveguide

from _logging import logger

from scipy.optimize import minimize

import numpy
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_target = 9.3e9
f_target_GHz = f_target / 1e9


def get_interpolator(E_data):
    return interp1d(E_data[:, 0].real, E_data[:, 1])


def calculate_signal_outgoing_mismatched_ports(  # nw: skrf.Network,
        S_matrix,
        # f,
        indexes_of_mismatched_ports,
        reflection_coefficients_of_mismatched_ports: numpy.ndarray,
        index_of_port_to_input_signal: int = 0,
        a_input_port=1 + 0j):
    """

    :param nw:
    :param f:
    :param indexes_of_mismatched_ports: shape (n,)
    :param reflection_coefficients_of_mismatched_ports: shape (n,), dtype=complex
    :return:
    """
    # indexes_of_mismatched_ports = numpy.array(indexes_of_mismatched_ports)
    # interpolater_nw = nw.interpolate([f])
    # bool_index = numpy.zeros(interpolater_nw.s.shape[1],dtype=bool)
    # bool_index[indexes_of_mismatched_ports] =True
    rows, columns = numpy.meshgrid(indexes_of_mismatched_ports, indexes_of_mismatched_ports, indexing='ij')
    # s_all_ports = s_matrix#nw.interpolate([f]).s[0]
    s = S_matrix[rows, columns]
    Gamma = numpy.matrix(numpy.diag(reflection_coefficients_of_mismatched_ports,
                                    ))  # .astype(complex)

    M = numpy.matrix(numpy.eye(*s.shape, )) - numpy.matrix(s) * Gamma
    si = numpy.matrix(S_matrix[indexes_of_mismatched_ports, index_of_port_to_input_signal]).T

    return numpy.linalg.inv(M) * si * a_input_port


def calculate_signals_outgoing(  # nw: skrf.Network,
        S_matrix,
        # f,
        indexes_of_mismatched_ports,
        reflection_coefficients_of_mismatched_ports: numpy.ndarray,
        index_of_port_to_input_signal: int = 0,
        a_input_port=1 + 0j):
    as_ = numpy.matrix(numpy.zeros(  # nw.nports,
        S_matrix.shape[0],
        dtype=complex)).T
    as_[index_of_port_to_input_signal] = a_input_port

    signal_outgoing_mismatched_ports = calculate_signal_outgoing_mismatched_ports(S_matrix, indexes_of_mismatched_ports,
                                                                                  reflection_coefficients_of_mismatched_ports,
                                                                                  index_of_port_to_input_signal).A.ravel()
    as_[indexes_of_mismatched_ports] = numpy.matrix(
        reflection_coefficients_of_mismatched_ports * signal_outgoing_mismatched_ports).T
    return numpy.matrix(S_matrix) * as_


if __name__ == '__main__':
    proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
        r"E:\CSTprojects\rfCompressor\cascadedHT\why_2hole\HplaneDualCrossLike-2c1o.cst",
        allow_interactive=True)
    S = {}
    key_data = 'data'
    key_interpolator = "interpolator"
    run_id = 0
    parameter_combination = proj_3D.get_3d().get_parameter_combination(run_id)
    port_ids = [1, 2, 3, 4, 5, ]
    for port_i in port_ids:
        S[port_i] = {}
        for port_j in port_ids:
            S_ij = numpy.array(
                proj_3D.get_3d().get_result_item('1D Results\\S-Parameters\\S%d,%d' % (port_i, port_j),
                                                 run_id).get_data())
            S[port_i][port_j] = {key_data: S_ij, key_interpolator: get_interpolator(S_ij)}

    f = S[1][1][key_interpolator].x
    s = numpy.zeros((len(f), len(port_ids), len(port_ids)), dtype=complex)
    z0 = numpy.zeros((len(f), len(port_ids)), dtype=complex)
    __name_map_from_skrf_to_CST = {
        i: port_id for i, port_id in enumerate(port_ids)
    }
    __name_map_from_CST_to_skrf = {
        __name_map_from_skrf_to_CST[key]: key for key in __name_map_from_skrf_to_CST
    }
    for port_i in port_ids:
        for port_j in port_ids:
            s[:, __name_map_from_CST_to_skrf[port_i], __name_map_from_CST_to_skrf[port_j]] = S[port_i][port_j][
                key_interpolator].y
            if port_i == port_j:
                z0[:, __name_map_from_CST_to_skrf[port_i]] = S[port_i][port_j][key_data][:, 2]

    nw = skrf.Network(frequency=skrf.Frequency.from_f(f, unit='GHz'), s=s, z0=z0)
    a1 = 1 + 0j
    # signal_outgoing_mismatched_ports = calculate_signal_outgoing_mismatched_ports(nw, f_target, [1, 4, 5],numpy.array( [-1, -1, -1],complex),
    #                                                                               0, a1).A.ravel()

    rect_WG_to_thru_arm = RectangularWaveguide(nw.frequency,  # z0_port = nw.z0[:,__name_map_from_CST_to_skrf[2]],
                                               a=30e-3, b=15e-3, mode_type='te', m=1, n=0)
    rect_WG_to_side_arms = RectangularWaveguide(nw.frequency,
                                                a=30e-3, b=15e-3, mode_type='te', m=1, n=0)
    beta_r_side_arms = numpy.linspace(0, 2 * numpy.pi, 100)
    beta_r_thru_arm = numpy.linspace(0, 2 * numpy.pi, 50)
    BETA_R_SIDE_ARM, BETA_R_THU_ARM, = numpy.meshgrid(beta_r_side_arms, beta_r_thru_arm,  # indexing='ij'
                                                      )
    # reflections =
    S_matrix_at_f_target = (nw.interpolate([f_target]).s[0])
    vectf_calculate_signals_outgoing = numpy.vectorize(
        lambda beta_r_thru_arm, beta_r_side_arms: calculate_signals_outgoing(S_matrix_at_f_target, [1, 3, 4],
                                                                             numpy.array([
                                                                                 1 - 2 / (1j * numpy.tan(
                                                                                     beta_r_thru_arm) + 1),
                                                                                 1 - 2 / (1j * numpy.tan(
                                                                                     beta_r_side_arms) + 1),
                                                                                 1 - 2 / (1j * numpy.tan(
                                                                                     beta_r_side_arms) + 1),
                                                                                 # rect_WG_to_thru_arm.delay_short(beta_r_thru_arm, unit='rad').interpolate([f_target]).s[0, 0, 0],
                                                                                 # rect_WG_to_side_arms.delay_short(beta_r_side_arms, unit='rad').interpolate([f_target]).s[0, 0, 0],
                                                                                 # rect_WG_to_side_arms.delay_short(beta_r_side_arms, unit='rad').interpolate([f_target]).s[0, 0, 0],
                                                                             ],
                                                                                 complex),
                                                                             0,
                                                                             a1).A.ravel(), signature="(),()->(n)")
    # vectf_calculate_signals_outgoing(0.*numpy.pi, 0.142 * numpy.pi)
    # vectf_calculate_signals_outgoing(0.4*numpy.pi, 0.1 * numpy.pi)

    SIGNAL_OUTs = vectf_calculate_signals_outgoing(BETA_R_THU_ARM, BETA_R_SIDE_ARM)
    TOTAL_OUTs = (numpy.abs(SIGNAL_OUTs[:, :, 2]) ** 2 #+ numpy.abs(SIGNAL_OUTs[:, :, 3]) ** 2
                  ) ** 0.5

    plt.figure(figsize=(4, 3), constrained_layout=True)
    d_total_out = 0.05
    # nw3.plot_s_mag()
    cf = plt.contourf(BETA_R_THU_ARM / numpy.pi, BETA_R_SIDE_ARM / numpy.pi, numpy.abs(TOTAL_OUTs),
                      levels=numpy.arange(0, 1.0 + d_total_out, d_total_out)
                      )
    plt.colorbar(cf)
    plt.xlabel(r"$(\beta r)_2 / \pi$")
    plt.ylabel(r"$(\beta r)_4 / \pi$")
    plt.title("OutDz1 = %d mm" % (parameter_combination["OutDz1"]))

    z0_WG_side_arm_interpolator = interp1d(f, rect_WG_to_side_arms.z0)
    beta_side_arm_interpolator = interp1d(f, rect_WG_to_side_arms.beta)


    # plt.figure()
    # plt.plot(beta_r_thru_arm,numpy.abs(get_total_leakage(0, beta_r_thru_arm)) )
    # plt.figure()
    # _get_new_nw(0., beta_side_arm_interpolator(f_target_GHz) * (11 + 7.6) * 1e-3).plot_s_mag()

    def get_beta_r_side_arm_to_minimize_and_maximize_o3(beta_r_thru_arm):
        # beta_r2 = 0
        nw2 = skrf.connect(nw, 1, rect_WG_to_thru_arm.delay_short(d=beta_r_thru_arm, unit='rad'), 0)
        ratio_power_outgoing_side_arm = numpy.max(numpy.abs(nw2.interpolate([f_target]).s[0, [2,3], 0]) ** 2)
        logger.info(
            "Power passing forward one of the side arms = %.2f%%" % (ratio_power_outgoing_side_arm * 100))
        P0 = 1e9
        P = P0 * ratio_power_outgoing_side_arm

        max_E_inside_switch_cav_when_no_out = 4 * (numpy.abs(z0_WG_side_arm_interpolator(f_target_GHz)) * P / (
                rect_WG_to_side_arms.a * rect_WG_to_side_arms.b)) ** 0.5

        max_E_inside_ESWG = 4 * (numpy.abs(z0_WG_side_arm_interpolator(f_target_GHz)) * P0 / (
                rect_WG_to_thru_arm.a * rect_WG_to_thru_arm.b)) ** 0.5
        max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG = max_E_inside_switch_cav_when_no_out / max_E_inside_ESWG
        logger.info(
            "When the power output is zero, maximum E field inside the side arm is %.2e V/m\n(assuming the power input from port 1 is 1 GW)" % (
                max_E_inside_switch_cav_when_no_out))

        res = minimize(lambda beta_r_side_arms: numpy.sum(
            numpy.abs(vectf_calculate_signals_outgoing(beta_r_thru_arm, beta_r_side_arms[0])[[2, ]]) ** 2),
                       [0, ], )
        logger.info("res = %s" % res)

        beta_r_side_arms_to_minimize_out = res.x % numpy.pi
        side_arm_delay_short_min_out = rect_WG_to_side_arms.delay_short(d=beta_r_side_arms_to_minimize_out, unit='rad')

        nw_minimize_power_out = skrf.connect(skrf.connect(nw2, 2, side_arm_delay_short_min_out, 0), 2,
                                             side_arm_delay_short_min_out, 0)

        beta_r_side_arms_to_minimize_out = res.x % numpy.pi

        res = minimize(lambda beta_r_side_arms: -numpy.sum(
            numpy.abs(vectf_calculate_signals_outgoing(beta_r_thru_arm, beta_r_side_arms[0])[[2, ]]) ** 2),
                       [0, ], )
        logger.info("res = %s" % res)

        beta_r_side_arms_to_maximize_out = res.x % numpy.pi
        side_arm_delay_short_max_out = rect_WG_to_side_arms.delay_short(d=beta_r_side_arms_to_maximize_out, unit='rad')

        nw_maximize_power_out = skrf.connect(skrf.connect(nw2,2, side_arm_delay_short_max_out, 0), 2,
                                             side_arm_delay_short_max_out, 0)
        # plt.figure()
        # nw_maximize_power_out.plot_s_mag()

        return (beta_r_side_arms_to_minimize_out, beta_r_side_arms_to_maximize_out,
                nw_minimize_power_out, nw_maximize_power_out,
                max_E_inside_switch_cav_when_no_out, max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG)


    arr_beta_r_side_arms_to_minimize_o3 = []
    arr_beta_r_side_arms_to_maximize_o3 = []
    arr_nw_minimize_power_out = []
    arr_nw_maximize_power_out = []
    arr_max_E_inside_switch_cav_when_no_out = []
    arr_max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG = []
    for beta_r2 in beta_r_thru_arm:
        beta_r4_to_minimize_o3, beta_r4_to_maximize_o3, nw_minimize_power_out, nw_maximize_power_out, max_E_inside_switch_cav_when_no_out, max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG = get_beta_r_side_arm_to_minimize_and_maximize_o3(
            beta_r2)
        arr_beta_r_side_arms_to_minimize_o3.append(beta_r4_to_minimize_o3)
        arr_beta_r_side_arms_to_maximize_o3.append(beta_r4_to_maximize_o3)
        arr_nw_minimize_power_out.append(nw_minimize_power_out)
        arr_nw_maximize_power_out.append(nw_maximize_power_out)
        arr_max_E_inside_switch_cav_when_no_out.append(max_E_inside_switch_cav_when_no_out)
        arr_max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG.append(
            max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG)
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 3), constrained_layout=True)
    axs[0].plot(beta_r_thru_arm / numpy.pi, arr_beta_r_side_arms_to_minimize_o3, label="min. power out")
    axs[0].plot(beta_r_thru_arm / numpy.pi, arr_beta_r_side_arms_to_maximize_o3,
                # numpy.array(arr_beta_r4_to_maximize_o3) % numpy.pi,
                label="max. power out")
    axs[0].legend()
    axs[1].plot(beta_r_thru_arm / numpy.pi, [10 * numpy.log10(
        numpy.sum(numpy.abs(arr_nw_minimize_power_out[i].interpolate([f_target]).s[0, [1,  ], 0]) ** 2)) for i in
                                             range(len(beta_r_thru_arm))])

    # plt.figure()
    # plt.plot(beta_r2s / numpy.pi, numpy.array(arr_max_E_inside_switch_cav_when_no_out) / 1e6,
    #          label="max_E_inside_switch_cav_when_no_out")
    # plt.xlabel(r"$(\beta r)_2 / \pi$")
    # plt.ylabel(r"$E_{max}$ (MV/m)")
    # plt.legend()

    # plt.figure()
    #
    # for i, beta_r2 in enumerate(beta_r2s):
    #     if i % 5 == 0:
    #         plt.plot(arr_nw_maximize_power_out[i].f / 1e9, numpy.abs(arr_nw_maximize_power_out[i].s[:, 1, 0]),
    #                  label=r"$(\beta r)_2 / \pi$ = %.2f" % (beta_r2 / numpy.pi))
    # plt.xlabel(r"$(\beta r)_2 / \pi$")
    # plt.legend()

    BETA_R_THU_ARM, FREQS_GHz = numpy.meshgrid(beta_r_thru_arm, f, indexing='ij')
    S_out_port1 = numpy.zeros((*FREQS_GHz.shape, 2), dtype=complex)
    for i, beta_r2 in enumerate(beta_r_thru_arm):
        S_out_port1[i, :] = arr_nw_maximize_power_out[i].s[:, [1, ], 0]
    d_leakage = 0.05
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 3), constrained_layout=True)
    cf = axs[0].contourf(BETA_R_THU_ARM / numpy.pi, FREQS_GHz, numpy.sum(numpy.abs(S_out_port1) ** 2, axis=-1),
                         levels=numpy.arange(0, 1.0 + d_leakage, d_leakage))
    c_ = axs[0].contour(BETA_R_THU_ARM / numpy.pi, FREQS_GHz, numpy.sum(numpy.abs(S_out_port1) ** 2, axis=-1),
                        [0.5, ])
    plt.clabel(c_, inline=True, fontsize=10)
    fig.colorbar(cf, ax=axs)
    axs[0].set_xlabel(r"$(\beta r)_2 / \pi$")
    axs[0].set_ylabel(r"frequency / GHz")

    axs[1].plot(beta_r_thru_arm / numpy.pi,
                numpy.array(arr_max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG) ** 2, )
    axs[1].set_xlabel(r"$(\beta r)_2 / \pi$")
    axs[1].set_ylabel(r"$\left(E_{max}^{switch}/E_{max}\right)^2$")
    plt.ylim(0, 1)
    # axs[1].legend()
    fig.suptitle("SCcenterDz = %d mm" % (parameter_combination["SCcenterDz"]))

    best_beta_r_thru_arm = 0.53 * numpy.pi
    (beta_r_side_arms_to_minimize_out, beta_r_side_arms_to_maximize_out,
     nw_minimize_power_out, nw_maximize_power_out,
     max_E_inside_switch_cav_when_no_out, max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG) =     get_beta_r_side_arm_to_minimize_and_maximize_o3(best_beta_r_thru_arm)
    # best_side_arm_min_out = rect_WG_to_thru_arm.delay_short(d=best_beta_r_side_arm_min_out * numpy.pi, unit='rad')
    plt.figure()
    plt.plot(nw_minimize_power_out.f/ 1e9,10 * numpy.log10(1-numpy.abs( nw_minimize_power_out.s[:,0,0])**2) ,label = "my calc")
    # nw_minimize_power_out.plot_s_db(m =  None, n = 0)
    plt.legend()

    plt.figure()
    plt.plot(nw_maximize_power_out.f/ 1e9, (1-numpy.abs( nw_maximize_power_out.s[:,0,0])**2) ,label = "my calc")
    # nw_minimize_power_out.plot_s_db(m =  None, n = 0)
    plt.legend()