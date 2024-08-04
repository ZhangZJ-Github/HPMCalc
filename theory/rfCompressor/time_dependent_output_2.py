# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 18:56
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : time_dependent_output.py
# @Software: PyCharm

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import scipy.constants as C
from scipy.fft import ifft
from scipy.optimize import curve_fit
from shapely.geometry import LineString

from _logging import logger

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_target = 9.3e9


def get_ts(signal: numpy.ndarray, t_start, dt):
    N = len(signal)
    return numpy.linspace(t_start, t_start + N * dt, N)


def to_df(o23, period_to_calculate_avg=2 / (f_target / 1e9)):
    df_o23 = pandas.DataFrame(o23)
    df_o23[key_period_for_calculate_avg] = df_o23[0] // period_to_calculate_avg
    df_o23[key_square] = df_o23[1] ** 2
    df_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
    df_o23[key_interpolated_periodic_avg_square] = numpy.interp(df_o23[0], df_time_avg[0], df_time_avg['square'], )
    df_o23[key_complex] = scipy.signal.hilbert(df_o23[1])
    return df_o23


class S_param_network:
    @staticmethod
    def get_S_paramter_interpolator(S23_3D):
        # return lambda f: numpy.interp(numpy.abs(f), numpy.real(S23_3D[:, 0]),
        #                                                          S23_3D[:, 1], left=complex(0), right=complex(0))
        return lambda f: numpy.piecewise(f, [numpy.real(f) >= 0, ],
                                         [lambda f: numpy.interp(numpy.real(f), numpy.real(S23_3D[:, 0]),
                                                                 S23_3D[:, 1], left=complex(0), right=complex(0)),
                                          lambda f: numpy.interp(-numpy.real(f), numpy.real(S23_3D[:, 0]),
                                                                 numpy.conj(S23_3D[:, 1]), left=complex(0),
                                                                 right=complex(0))
                                          ])

    def __init__(self, S_param_data: numpy.ndarray, impulse_response_duration, impulse_response_dt):
        """
        :param S_param_data: shape (N,2), 其中第一列为频率
        """
        self.S_param_data = S_param_data
        self.S_interpolator = self.get_S_paramter_interpolator(S_param_data)
        self.get_impulse_response(impulse_response_duration, impulse_response_dt)
        # self.impulse_response, self.dt = self.get_impulse_response(dt_to_get_impulse_response)

    def get_impulse_response(self, duration, dt):
        freqs = scipy.fftpack.fftfreq(int(duration // dt), dt)
        Nfreqs = len(freqs)
        self.impulse_response = impulse_response = ifft(self.S_interpolator(freqs.astype(complex)))[:Nfreqs // 2]
        self.dt = dt
        return impulse_response


class OM_2port:
    """
    2端口输出模块
    """

    def __init__(self, S11: S_param_network, S21: S_param_network):
        """
        输出模块，其中1端口为模块的输入端
        单位默认为SI标准单位 (m, Hz, s)
        """
        self.S11 = S11
        self.S21 = S21


class Compressor:
    def __init__(self, om_charging: OM_2port, om_discharging: OM_2port):
        self.om_charging = om_charging
        self.om_discharging = om_discharging
        assert self.om_charging.S21.dt == self.om_discharging.S21.dt
        self.dt = self.om_charging.S21.dt

    @staticmethod
    def get_ts(signal: numpy.ndarray, t_start, dt):
        N = len(signal)
        return numpy.linspace(t_start, t_start + N * dt, N)

    def run(self, f_target, t_charging_start, t_discharging_start, tend, Dt_MESS, ):
        t0 = t_charging_start
        dt = self.dt

        _get_ts = lambda signal: get_ts(signal, t0, dt)

        t = numpy.arange(t0, t_discharging_start, dt)

        i3_charging = numpy.piecewise(t, [(t > t0  # -10e-9
                                           ) & (t < t_discharging_start), ],
                                      [lambda t: numpy.sin(2 * numpy.pi * f_target * t), 0])

        o33_from_convolve_charging = scipy.signal.convolve(i3_charging, self.om_charging.S11.impulse_response)
        o33 = o33_from_convolve_charging
        while t[-1] < tend:
            i3_discharging = numpy.piecewise(t, [t > t_discharging_start, ],
                                             [lambda t: numpy.interp(t - Dt_MESS, _get_ts(o33), o33), 0])
            o33_from_convolve_discharging = scipy.signal.convolve(i3_discharging,
                                                                  self.om_discharging.S11.impulse_response, )
            if len(o33_from_convolve_discharging) - len(o33_from_convolve_charging) >= 0:
                o33 = numpy.pad(o33_from_convolve_charging,
                                (0, len(o33_from_convolve_discharging) - len(o33_from_convolve_charging)), 'constant',
                                constant_values=(0, 0)) + o33_from_convolve_discharging
                # logger.info("len(o33_from_convolve_discharging) - len(o33_from_convolve_charging) > 0")
            else:

                o33 = o33_from_convolve_charging[:len(o33_from_convolve_discharging)] + o33_from_convolve_discharging
                # logger.info("else")
            t = numpy.arange(t[0], t[-1] + Dt_MESS, dt)

        o23_charging = scipy.signal.convolve(i3_charging, self.om_charging.S21.impulse_response)
        o23_discharging = scipy.signal.convolve(i3_discharging, self.om_discharging.S21.impulse_response)
        o23 = numpy.pad(o23_charging,
                        (0, len(o23_discharging) - len(o23_charging)), 'constant',
                        constant_values=(0, 0)) + o23_discharging
        df_i3_charging = to_df(numpy.vstack((_get_ts(i3_charging), i3_charging)).T.astype(float), 2 / f_target)
        df_i3_discharging = to_df(numpy.vstack((_get_ts(i3_discharging), i3_discharging)).T.astype(float), 2 / f_target)
        df_o33 = to_df(numpy.vstack((get_ts(o33, t0, dt), o33.real)).T.astype(float), 2 / f_target)
        df_o23 = to_df(numpy.vstack((get_ts(o23, t0, dt), o23.real)).T.astype(float), 2 / f_target)

        return df_i3_charging, df_i3_discharging, df_o33, df_o23

    def correct_Dt_MESS(self, f_target, Dt_MESS):
        """
        修正相位：在一个RF周期内微调，使其满足S参数的约束，而几乎不影响幅值。
        :param f_target:
        :param Dt_MESS:
        :return:
        """
        angle_S33_charging = numpy.angle(self.om_charging.S11.S_interpolator(complex(f_target)))  # +0.7
        T_RF = (1 / (f_target))
        # Dt_lefts = numpy.array((*numpy.arange(0.2, 1, 0.2), *numpy.arange(1, 10, 1), 20))
        return (Dt_MESS // T_RF) * T_RF + angle_S33_charging / (2 * numpy.pi) * T_RF


if __name__ == '__main__':
    # 需包含放能阶段的S参数
    proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
        r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch_2-1.cst",
        allow_interactive=True)
    proj_3D_charging: cst.results.ProjectFile = cst.results.ProjectFile(
        r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch_2-1.ES.cst",
        allow_interactive=True)
    S33_3D = numpy.array(proj_3D.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3', ).get_data())
    S23_3D = numpy.array(proj_3D.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3', ).get_data())

    S33_3D_charging = numpy.array(
        proj_3D_charging.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3', ).get_data())
    S23_3D_charging = numpy.array(
        proj_3D_charging.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3', ).get_data())

    impulse_duration = 50.0
    impulse_dt = 1 / (f_target / 1e9) / 20
    compressor = Compressor(
        OM_2port(S_param_network(S33_3D_charging, impulse_duration, impulse_dt),
                 S_param_network(S23_3D_charging, impulse_duration, impulse_dt)),
        OM_2port(S_param_network(S33_3D, impulse_duration, impulse_dt),
                 S_param_network(S23_3D, impulse_duration, impulse_dt)), )

    Dt_lefts = compressor.correct_Dt_MESS(
        f_target / 1e9,
        numpy.array((*numpy.arange(0.2, 1, 0.2), *numpy.arange(1, 10, 1), 20)))
    recorded_o23 = []

    t_charging_start = -2 * max(max(Dt_lefts), impulse_duration)
    t_discharging_start = 0.0
    tend = 40.0
    for i, Dt_left in enumerate(Dt_lefts):
        df_i3_charging, df_i3_discharging, df_o33, df_o23 = compressor.run(f_target / 1e9, t_charging_start, 0, tend,
                                                                           Dt_left)
        recorded_o23.append(df_o23)
        logger.info("%d, %.2f ns" % (i, Dt_left))

    trusted_signal = lambda df: (df[0] > t_discharging_start) & (df[1] < tend)
    eta_OM_max = []

    for i, df_o23 in enumerate(recorded_o23):
        eta_OM_max.append(2 * df_o23[trusted_signal(df_o23)][key_interpolated_periodic_avg_square].max())
    eta_OM_max = numpy.array(eta_OM_max)

    plt.figure()
    plt.plot(df_i3_discharging[0], (df_i3_discharging[1]), label='i3')
    plt.plot(df_o33[0], (df_o33[1]), label='o33')
    plt.plot(df_o23[0], (df_o23[1]), label='o23', lw=5)
    plt.legend()

    plt.figure()
    plt.plot(df_i3_discharging[0], numpy.abs(df_i3_discharging[key_complex]) ** 2, label='i3')
    plt.plot(df_o33[0], numpy.abs(df_o33[key_complex]) ** 2, label='o33')
    plt.plot(df_o23[0], numpy.abs(df_o23[key_complex]) ** 2, label='o23', lw=5)
    plt.legend()

    plt.figure()
    plt.plot(df_i3_discharging[0], (df_i3_discharging[key_interpolated_periodic_avg_square]) * 2, label='i3')
    plt.plot(df_o33[0], (df_o33[key_interpolated_periodic_avg_square]) * 2, label='o33')
    plt.plot(df_o23[0], (df_o23[key_interpolated_periodic_avg_square]) * 2, label='o23', lw=5)
    plt.legend()

    gammadata = numpy.array(
        proj_3D_charging.get_3d().get_result_item('1D Results\\Port Information\\Gamma\\3(1)').get_data())
    v_p = 2 * numpy.pi * f_target / numpy.interp((f_target / 1e9), gammadata[:, 0].real, gammadata[:, 1]).imag
    v_g = C.c ** 2 / v_p

    plt.figure(figsize=(5, 4), constrained_layout=True)
    for i, df_o23_ in enumerate(recorded_o23):
        if i % 2 == 0 or False:
            # plt.plot(df_o23_[0],(numpy.abs(df_o23_[key_complex])**2),label = '$\Delta t$ = %.1f ns ($L_1$ = %.2f m)'%(Dt_lefts[i]/1e-9, Dt_lefts[i]*v_g/2))
            plt.plot(df_o23_[0], df_o23_[key_interpolated_periodic_avg_square] * 2,
                     label='$\Delta t$ = %.2f ns ($L_1$ = %.2f m)' % (Dt_lefts[i], Dt_lefts[i] * 1e-9 * v_g / 2))
            # plt.plot(df_o23_[0],(numpy.abs(df_o23_[key_complex])**2),label = '$L_1$ = %.2f m'%( Dt_lefts[i]*v_g/2))
    plt.xlabel('time (ns)')
    plt.ylabel(r'$\eta_{OM}(t)$')
    # plt.ylabel(r'normalized amplitude / $\sqrt{W}$')
    plt.legend()
    plt.savefig("eta_OM_of_different_Dt.eps")

    G_OM = 1 / (1 - (numpy.abs(compressor.om_charging.S21.S_interpolator(complex(f_target / 1e9))) ** 2 + numpy.abs(
        compressor.om_charging.S11.S_interpolator(complex(f_target / 1e9))) ** 2))
    L_OM = (299.77 - 20) * 1e-3


    def func_G_cav_peak(Dt_lefts):
        return G_OM * 2 * L_OM / v_g / (Dt_lefts + 2 * L_OM / v_g)


    G_cav_peak = func_G_cav_peak(Dt_lefts * 1e-9)
    _func_eta_OM_peak = lambda Dt_left, eta_inf, tau, Dt_right: numpy.piecewise(
        Dt_left, [Dt_left + Dt_right > 0],
        (lambda Dt_left: eta_inf * (1 - numpy.exp(-(Dt_left + Dt_right) / tau)) ** 2, 0))
    df = pandas.read_csv(
        r"F:\changeworld\HPMCalc\theory\rfCompressor\test_SES_switch01.1.cst.results\0218\signals\o23.csv")

    t_ = numpy.arange(0.0, 40.0, compressor.dt)
    test_01 = scipy.signal.convolve(compressor.om_discharging.S21.impulse_response,
                                    numpy.sin(2 * numpy.pi * f_target / 1e9 * t_))
    df_test_01 = to_df(numpy.array((t_, test_01[:len(t_)])).T.real)
    # df_test_01_interpolator = scipy.interpolate.interp1d
    sig_o23 = numpy.array(
        proj_3D.get_schematic().get_result_item('Tasks\\Tran1\\TD Signals\\O2,3', ).get_data())

    out_pulse_duration = []
    # arr_intersected_pts =[]
    for i, df_o23_ in enumerate(recorded_o23):
        df_o23_ = df_o23_[trusted_signal(df_o23_)]
        eta_OM = ((df_o23_)[key_interpolated_periodic_avg_square] * 2)
        ls = LineString(numpy.array((df_o23_[0], eta_OM.values)).T)
        half_maximum = eta_OM.max() / 2
        ls2 = LineString(numpy.array([[min(df_o23[0]), half_maximum],
                                      [max(df_o23[0]), half_maximum]]))
        intersected_pts = [pt for pt in ls.intersection(ls2).geoms]
        intersected_pts.sort(key=lambda pt: pt.x)
        possible_distance = [intersected_pts[i].distance(intersected_pts[i + 1]) for i in
                             range(len(intersected_pts) - 1)]
        out_pulse_duration.append(max(possible_distance))
        # out_pulse_duration.append(intersected_pts[-1].distance(intersected_pts[-2]))
        # shapely.lib.intersection_all([ls,ls2])
        # arr_intersected_pts.append(intersected_pts)

    out_pulse_duration = numpy.array(out_pulse_duration)

    (eta_inf, tau, Dt_right), cov = curve_fit(_func_eta_OM_peak, Dt_lefts, eta_OM_max, p0=[0.88, 3, 0.5])
    fig, axs = plt.subplots(4, 1, figsize=(4, 6), constrained_layout=True, sharex=True)
    axs[0].plot(Dt_lefts, out_pulse_duration, '.', label="output pulse duration")
    axs[0].set_ylabel("duration (ns)")
    _func_out_pulse_duration = lambda Dt_left, a, Dt_OM: a * Dt_left + Dt_OM
    __filter = Dt_lefts > 1.0
    (a, Dt_OM2,), cov = curve_fit(_func_out_pulse_duration, Dt_lefts[__filter], out_pulse_duration[__filter])
    __Dt_lefts = numpy.linspace(-0., max(Dt_lefts), 1000)
    axs[0].plot(__Dt_lefts, _func_out_pulse_duration(__Dt_lefts, a, Dt_OM2), '--',
                label=r"$%.2f (\Delta t + %.2f~\rm ns)$" % (a, Dt_OM2 / a))
    axs[1].plot(Dt_lefts, eta_OM_max, '.', label="$\eta_{OM, peak}$")
    axs[1].plot(__Dt_lefts, _func_eta_OM_peak(__Dt_lefts, eta_inf, tau, Dt_right), '--',
                label=r"$%.3f [1-exp(-\frac{t+ %.2f\ \rm{ns}}{%.2f\ \rm{ns}})]^2$" % (eta_inf, Dt_right, tau,))
    # axs[0].plot(df_test_01[0],df_test_01[key_interpolated_periodic_avg_square]*2 ,label = 'convolve')

    (P23_inf, tau2, Dt_right2) = [0.7254, tau, -0.9]
    # (P23, tau2, Dt_right2),cov = curve_fit(   _func_eta_OM_peak, df_test_01[0],df_test_01[key_interpolated_periodic_avg_square]*2,p0= (P23, tau2, Dt_right2) )
    logger.info((P23_inf, tau2, Dt_right2))
    # axs[0].plot(df_test_01[0], _func_eta_OM_peak(df_test_01[0].values, P23_inf, tau2, Dt_right2), label ='convolve, fitted')
    _func_eta_OM_peak2 = lambda t, Dt: numpy.interp((t + Dt), df_test_01[0],
                                                    2 * df_test_01[key_interpolated_periodic_avg_square])
    (Dt_OM,), cov = curve_fit(_func_eta_OM_peak2, Dt_lefts, eta_OM_max, p0=[1])
    # Dt_OM = 1e-9
    # axs[0].plot(1e9*Dt_lefts,_func_eta_OM_peak2(Dt_lefts,Dt_OM),label = "Dt_OM = %.2f ns"%(1e9*Dt_OM))
    # axs[0].plot(df['time/ns'], df['signal']**2)

    axs[2].plot(__Dt_lefts, func_G_cav_peak(__Dt_lefts * 1e-9), label='$G_{cav,peak}$')

    axs[3].plot(Dt_lefts, eta_OM_max * G_cav_peak, '.-', label="$G_{OM,peak}$")
    # axs[3].plot(1e9*__Dt_lefts, _func_eta_OM_peak(__Dt_lefts,eta_inf, tau, Dt_right)*func_G_cav_peak(__Dt_lefts),'.-',label  = "$G_{OM,peak}$")

    for ax in axs: ax.legend()
    plt.xlabel("$\Delta t$ (ns)")
    plt.savefig("G_vs_Delta_t.eps")

    ts = numpy.arange(0, 100, compressor.dt)
    sin = numpy.sin(2 * numpy.pi * f_target / 1e9 * ts)
    plt.figure()
    plt.plot(ts, sin)
    signal = scipy.signal.convolve(sin, compressor.om_discharging.S21.impulse_response)
    plt.plot(get_ts(signal, ts[0], ts[1] - ts[0]), signal)

    plt.figure()
    plt.plot(get_ts(compressor.om_discharging.S21.impulse_response, 0, compressor.dt),
             compressor.om_discharging.S21.impulse_response)
