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


def get_beta_r4_to_eliminate_o3(S31, S41, S34, S44
                                ):
    """
    问题描述：
    对于包含1, 3, 4三端口的网络S，
    1端口输入，
    3端口接匹配负载（输出），
    4端口（支臂）接长度为r4的波导，且终端短路，
    求beta * r4的值，使3端口输出为0
    其中 beta = 2 pi / lambda_g 为支臂波导的纵向波数

    :param S_II_34:
    :return:
    """
    Gamma = S31 / (S31 * S44 - S34 * S41)
    beta_r4 = numpy.arctan((1 + Gamma) / (1j * (1 - Gamma)))
    return beta_r4


def get_o3(S31, S41, S34, S44, Gamma, i1=1 + 0j):
    return (S31 + Gamma * S34 * S41 / (1 - S44 * Gamma)) * i1


def get_interpolator(E_data):
    return interp1d(E_data[:, 0].real, E_data[:, 1])


if __name__ == '__main__':
    proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
        r"E:\CSTprojects\rfCompressor\cascadedHT\why_2hole\HplaneCrossLike.cst",
        allow_interactive=True)
    S = {}
    key_data = 'data'
    key_interpolator = "interpolator"
    run_id = 0
    parameter_combination = proj_3D.get_3d().get_parameter_combination(run_id)
    port_ids = [1, 2, 3, 4]
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

    rect_WG_to_port2 = RectangularWaveguide(nw.frequency,  # z0_port = nw.z0[:,__name_map_from_CST_to_skrf[2]],
                                            a=30e-3, b=15e-3, mode_type='te', m=1, n=0)
    rect_WG_to_port4 = RectangularWaveguide(nw.frequency,
                                            a=30e-3, b=15e-3, mode_type='te', m=1, n=0)
    z0_port4_interpolator = interp1d(f, rect_WG_to_port4.z0)
    beta_r2s = numpy.linspace(0, 1 * numpy.pi, 50)
    beta_r4s = numpy.linspace(0, 1 * numpy.pi, 50)


    def _get_new_nw(beta_r2, beta_r4):
        nw2 = skrf.connect(nw, 1, rect_WG_to_port2.delay_short(d=beta_r2, unit='rad'), 0)
        nw3 = skrf.connect(nw2, 2, rect_WG_to_port4.delay_short(d=beta_r4, unit='rad'), 0)
        return nw3


    def _get_leakage(beta_r2, beta_r4):
        return _get_new_nw(beta_r2, beta_r4).interpolate([f_target]).s[0, 1, 0]


    get_leakage = numpy.vectorize(_get_leakage, )
    d_leakage = 0.05
    BETA_R2, BETA_R4 = numpy.meshgrid(beta_r2s, beta_r4s)


    if 0:
        LEAKAGES = get_leakage(BETA_R2, BETA_R4)
        plt.figure(figsize=(4, 3), constrained_layout=True)
        # nw3.plot_s_mag()
        cf = plt.contourf(BETA_R2 / numpy.pi, BETA_R4 / numpy.pi, numpy.abs(LEAKAGES),
                          levels=numpy.arange(0, 1.0 + d_leakage, d_leakage))
        plt.colorbar(cf)
        plt.xlabel(r"$(\beta r)_2 / \pi$")
        plt.ylabel(r"$(\beta r)_4 / \pi$")
        plt.title("OutDz1 = %d mm" % (parameter_combination["OutDz1"]))

    beta_4_interpolator = interp1d(f, rect_WG_to_port4.beta)


    # plt.figure()
    # plt.plot(beta_r4s,numpy.abs(get_leakage(0, beta_r4s)) )
    # plt.figure()
    # _get_new_nw(0., beta_4_interpolator(f_target_GHz) * (11 + 7.6) * 1e-3).plot_s_mag()

    def get_beta_r4_to_minimize_and_maximize_o3(beta_r2):
        # beta_r2 = 0
        nw2 = skrf.connect(nw, 1, rect_WG_to_port2.delay_short(d=beta_r2, unit='rad'), 0)
        S_port4_port1 = numpy.abs(nw2.interpolate([f_target]).s[0, 2, 0])
        logger.info(
            "Power passing forward side arm = %.2f%%" % (S_port4_port1 ** 2 * 100))
        P0 = 1e9
        P = P0 * S_port4_port1 ** 2

        max_E_inside_switch_cav_when_no_out = 4 * (numpy.abs(z0_port4_interpolator(f_target_GHz)) * P / (
                rect_WG_to_port4.a * rect_WG_to_port4.b)) ** 0.5

        max_E_inside_ESWG = 4 * (numpy.abs(z0_port4_interpolator(f_target_GHz)) * P0 / (
                rect_WG_to_port2.a * rect_WG_to_port2.b)) ** 0.5
        max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG = max_E_inside_switch_cav_when_no_out / max_E_inside_ESWG
        logger.info(
            "When the power output is zero, maximum E field inside the side arm is %.2e V/m\n(assuming the power input from port 1 is 1 GW)" % (
                max_E_inside_switch_cav_when_no_out))

        beta_r4_to_minimize_o3 = get_beta_r4_to_eliminate_o3(nw2.interpolate([f_target]).s[0, 1, 0,],
                                                             nw2.interpolate([f_target]).s[0, 2, 0,],
                                                             nw2.interpolate([f_target]).s[0, 1, 2,],
                                                             nw2.interpolate([f_target]).s[0, 2, 2,], ).real
        nw_minimize_power_out = skrf.connect(nw2, 2, rect_WG_to_port4.delay_short(d=beta_r4_to_minimize_o3, unit='rad'),
                                             0)
        # plt.figure()
        # nw_minimize_power_out.plot_s_mag()
        # res = minimize(lambda beta_r4:-numpy.abs(get_o3(nw2.interpolate([f_target]).s[0, 1, 0,],
        #                                                       nw2.interpolate([f_target]).s[0, 2, 0,],
        #                                                       nw2.interpolate([f_target]).s[0, 1, 2,],
        #                                                       nw2.interpolate([f_target]).s[0, 2, 2,], 1j * numpy.tan(beta_r4[0]))),[0,],)
        res = minimize(lambda beta_r4: - numpy.abs(
            skrf.connect(nw2, 2, rect_WG_to_port4.delay_short(d=beta_r4, unit='rad'), 0).interpolate([f_target]).s[
                0, 1, 0]),
                       [0, ], )
        logger.info("res = %s" % res)
        beta_r4_to_maximize_o3 = res.x % numpy.pi

        nw_maximize_power_out = skrf.connect(nw2, 2, rect_WG_to_port4.delay_short(d=beta_r4_to_maximize_o3, unit='rad'),
                                             0)
        # plt.figure()
        # nw_maximize_power_out.plot_s_mag()
        return beta_r4_to_minimize_o3, beta_r4_to_maximize_o3, nw_minimize_power_out, nw_maximize_power_out, max_E_inside_switch_cav_when_no_out, max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG


    arr_beta_r4_to_minimize_o3 = []
    arr_beta_r4_to_maximize_o3 = []
    arr_nw_minimize_power_out = []
    arr_nw_maximize_power_out = []
    arr_max_E_inside_switch_cav_when_no_out = []
    arr_max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG = []
    for beta_r2 in beta_r2s:
        beta_r4_to_minimize_o3, beta_r4_to_maximize_o3, nw_minimize_power_out, nw_maximize_power_out, max_E_inside_switch_cav_when_no_out, max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG = get_beta_r4_to_minimize_and_maximize_o3(
            beta_r2)
        arr_beta_r4_to_minimize_o3.append(beta_r4_to_minimize_o3)
        arr_beta_r4_to_maximize_o3.append(beta_r4_to_maximize_o3)
        arr_nw_minimize_power_out.append(nw_minimize_power_out)
        arr_nw_maximize_power_out.append(nw_maximize_power_out)
        arr_max_E_inside_switch_cav_when_no_out.append(max_E_inside_switch_cav_when_no_out)
        arr_max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG.append(
            max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG)
    # plt.figure()
    # plt.plot(beta_r2s / numpy.pi, arr_beta_r4_to_minimize_o3, label="min. power out")
    # plt.plot(beta_r2s / numpy.pi, numpy.array(arr_beta_r4_to_maximize_o3) % numpy.pi, label="max. power out")
    # plt.legend()

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

    BETA_R2, FREQS_GHz = numpy.meshgrid(beta_r2s, f, indexing='ij')
    S_out_port1 = numpy.zeros(FREQS_GHz.shape, dtype=complex)
    for i, beta_r2 in enumerate(beta_r2s):
        S_out_port1[i, :] = arr_nw_maximize_power_out[i].s[:, 1, 0]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 3), constrained_layout=True)
    cf = axs[0].contourf(BETA_R2 / numpy.pi, FREQS_GHz, numpy.abs(S_out_port1)**2,
                         levels=numpy.arange(0, 1.0 + d_leakage, d_leakage))
    c_ =axs[0].contour(BETA_R2 / numpy.pi, FREQS_GHz, numpy.abs(S_out_port1)**2,
                    [0.5,])
    plt.clabel(c_, inline=True, fontsize=10)
    fig.colorbar(cf, ax=axs)
    axs[0].set_xlabel(r"$(\beta r)_2 / \pi$")
    axs[0].set_ylabel(r"frequency / GHz")

    axs[1].plot(beta_r2s / numpy.pi, numpy.array(arr_max_E_inside_switch_cav_when_no_out_to_max_E_inside_ESWG) **2, )
    axs[1].set_xlabel(r"$(\beta r)_2 / \pi$")
    axs[1].set_ylabel(r"$\left(E_{max}^{switch}/E_{max}\right)^2$")
    plt.ylim(0, 1)
    # axs[1].legend()
    fig.suptitle("OutDz1 = %d mm" % (parameter_combination["OutDz1"]))
