# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 18:56
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : time_dependent_output.py
# @Software: PyCharm
import enum
import typing

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import skrf
from scipy.fft import ifft
from skrf.network import Network

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from _logging import logger


def convergent_convolve(*args, **kwargs):
    return scipy.signal.convolve(*args, **kwargs) / 1.028 * 0.995


convolve = scipy.signal.convolve  # convergent_convolve# or scipy.signal.convolve


# convolve = convergent_convolve# or scipy.signal.convolve
def get_S_parameter_interpolator(nw: Network):
    # assert numpy.all(nw.f>0)
    # return lambda f: numpy.piecewise(f, [f >= 0, ], [
    #     interp1d(nw.f, nw.s, axis=0, fill_value=0.j, bounds_error=False, assume_sorted=True)(f),
    #     (interp1d(nw.f, nw.s, axis=0, fill_value=0.j, bounds_error=False, assume_sorted=True)(-f)).conj(),
    # ])

    interpolator = interp1d(numpy.array([*(-nw.f[::-1]), *nw.f]), numpy.array([*nw.s.conj()[::-1], *nw.s]), axis=0,
                            fill_value=0.j, bounds_error=False, assume_sorted=True)

    # return  lambda f: interpolator(f) + interpolator(-f).conj()
    return interpolator


plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_ref = 9.3e9


# def get_ts(signal: numpy.ndarray, t_start, dt):
#     N = len(signal)
#     return numpy.linspace(t_start, t_start + N * dt, N)
def extend_time_sequence(ts, signal: numpy.ndarray, ):
    dt = ts[1] - ts[0]
    N = len(signal)
    return numpy.linspace(ts[0], ts[0] + N * dt, N)


def to_df(o23, period_to_calculate_avg=2 / (f_ref / 1e9)):
    df_o23 = pandas.DataFrame(o23)
    df_o23[key_period_for_calculate_avg] = df_o23[0] // period_to_calculate_avg
    df_o23[key_square] = df_o23[1] ** 2
    df_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
    df_o23[key_interpolated_periodic_avg_square] = numpy.interp(df_o23[0], df_time_avg[0], df_time_avg['square'], )
    df_o23[key_complex] = scipy.signal.hilbert(df_o23[1])
    return df_o23


def get_impulse_response(get_S_parameter, ts, axis=0):
    """

    :param get_S_parameter:
    :param ts: uniform-spaced time sequence
    :return:
    """
    N = len(ts)  # //2
    dt = ts[1] - ts[0]
    freqs = scipy.fftpack.fftfreq(N, dt)

    impulse_response = ifft(get_S_parameter(freqs), axis=axis)

    return numpy.arange(0, (N) * dt, dt) - (N - N // 2) * dt, numpy.array(
        (*impulse_response[N // 2:], *impulse_response[:N // 2],)).real


# class MyNetwork(Network):
#     def __init__(self, *args_for_Network, **kwargs ):
#         super(MyNetwork, self).__init__(*args_for_Network,**kwargs)
#
#         self.S_parameter_interpolator =  get_S_parameter_interpolator(self)
#         self.cached_impulse_response =(None,None)# numpy.array([]),numpy.array([])
#         time_seq_for_generate_impulse_response =kwargs.get("time_seq_for_generate_impulse_response", None)
#         if time_seq_for_generate_impulse_response  :
#             self.cache_impulse_response (time_seq_for_generate_impulse_response)
#     def cache_impulse_response(self, time_seq_for_generate_impulse_response :numpy.ndarray):
#         self.cached_impulse_response =        get_impulse_response(self.S_parameter_interpolator,time_seq_for_generate_impulse_response)

from typing import TypeVar


class Compressor:
    class Regime(enum.Enum):
        charging = 0
        discharging = 1

    Self = TypeVar("Self", bound="Compressor")

    def __init__(self, nw_discharging: Network, nw_charging: Network, ):
        self.nw_discharging = nw_discharging
        self.nw_charging = nw_charging
        self.networks = {
            self.Regime.charging.name: self.nw_charging,
            self.Regime.discharging.name: self.nw_discharging,
        }
        # self.ts_and_impulse_response = get_impulse_response(  get_S_parameter_interpolator(compressor.nw_charging),ts)

        self.f_ref = self.nw_discharging.f.max()

        self.cache_impulse_responses()

    def embed_with(self, network: Network):
        return Compressor(
            network ** self.nw_discharging,
            network ** self.nw_charging, )

    def embed_with_waveguide(self, CST_gamma_data_of_the_waveguide, L,  # ignore_attenuation
                             ):
        return self.embed_with(
            build_network_of_waveguide(CST_gamma_data_of_the_waveguide, self.nw_charging.f / 1e9, L))

    def deembed_with_waveguide(self, CST_gamma_data_of_the_waveguide, L):
        return self.embed_with(
            build_network_of_waveguide(CST_gamma_data_of_the_waveguide, self.nw_charging.f / 1e9, L).inv)

    def cache_impulse_responses(self,
                               ts: numpy.ndarray = None
                               # dt = None,
                               # impulse_response_duration =None
                               ):
        MIN_DURATION = 200e-9


        if ts is None:
            ts = numpy.arange(0, MIN_DURATION, 1 / self.f_ref / 5)

        self.dt = ts[1] - ts[0]
        if ts[-1]-ts[0]<MIN_DURATION:
            ts = numpy.arange(ts[0],ts[0]+MIN_DURATION+self.dt, ts[1]-ts[0], )

        self.impulse_response_time_seqs = {
            self.Regime.charging.name: get_impulse_response(get_S_parameter_interpolator(self.nw_charging), ts),
            self.Regime.discharging.name: get_impulse_response(get_S_parameter_interpolator(self.nw_discharging), ts)
        }
        self.impulse_response_time_seq_charging = self.impulse_response_time_seqs[
            self.Regime.charging.name]  # get_impulse_response(get_S_parameter_interpolator(self.nw_charging), ts)
        self.impulse_response_time_seq_discharging = self.impulse_response_time_seqs[
            self.Regime.discharging.name]  # get_impulse_response(get_S_parameter_interpolator(self.nw_discharging), ts)

    def get_impulse_response_time_seq_ij(self, i, j, regime_name=Regime.charging.name):

        # return {
        #     self.Regime.charging .name : (self.impulse_response_time_seq_charging[0], self.impulse_response_time_seq_charging[1][:, i, j],),
        #     self.Regime.discharging .name : (self.impulse_response_time_seq_discharging[0], self.impulse_response_time_seq_discharging[1][:, i, j],),
        # }[regime_name]
        response = self.impulse_response_time_seqs[regime_name]
        return response[0], response[1][:, i, j]

    @staticmethod
    def calculate_response_time_seq_ij(impulse_response_time_seq_ij: typing.Tuple[numpy.ndarray, numpy.ndarray],
                                       input_signal: typing.Tuple[numpy.ndarray, numpy.ndarray]):
        """
        :param impulse_response_time_seq_ij: 形如(t, y)，其中y是由Sij(f)转化而来的脉冲响应信号
        :param input_signal: (t, y)
        :return: (t, y)
        """

        convolved = convolve(impulse_response_time_seq_ij[1],
                             input_signal[1],
                             )[len(impulse_response_time_seq_ij[0]) // 2:]
        ts = extend_time_sequence(input_signal[0], convolved)  # / 1e-9,

        return (ts, convolved)

    def correct_Dt_MESS(self, f_target, Dt_MESS):
        """
        修正相位：在一个RF周期内微调，使其满足S参数的约束，而几乎不影响幅值。
        :param f_target:
        :param Dt_MESS:
        :return:
        """
        angle_ = numpy.angle(1 / self.nw_charging.interpolate([f_target]).s[0, 0, 0])
        T_RF = (1.0 / (f_target))
        # Dt_lefts = numpy.array((*numpy.arange(0.2, 1, 0.2), *numpy.arange(1, 10, 1), 20))
        # return (Dt_MESS // T_RF +1 + angle_S33_charging / (2 * numpy.pi) ) * T_RF
        # logger.info("angle_ = %.2f degree"%(numpy.rad2deg(angle_)))
        return (Dt_MESS // T_RF + 1 - angle_ / (2 * numpy.pi)) * T_RF

    @staticmethod
    def _add_signals_with_the_same_start_time(shorter, longer):
        return numpy.pad(shorter,
                         (0, len(longer) - len(shorter)), 'constant',
                         constant_values=(0, 0)) + longer

    def run(self, initial_input_time_seq: typing.Tuple[numpy.ndarray, numpy.ndarray],
            # nw_MESS:Network,
            # f_target, t_charging_start, t_discharging_start,            dt,
            tend, Dt_MESS,
            ):
        """
        :param initial_input_time_seq: (t, y(t))
        :param tend:
        :param Dt_MESS:
        :return:
        """

        t, i1_charging = initial_input_time_seq
        assert t[-1] - t[0] > Dt_MESS
        _get_ts = lambda signal: extend_time_sequence(t, signal)

        t_charging_start, t_discharging_start = t[0], t[-1]
        if numpy.abs((t[1]-t[0])/self.dt - 1)>1e-4:
            logger.info("%.2e, %.2e"%(t[1]-t[0], self.dt))
            self.cache_impulse_responses(t)
            logger.info("Impulse responses are re-cached.")


        o11_from_convolve_charging = self.calculate_response_time_seq_ij(
            self.get_impulse_response_time_seq_ij(0, 0, self.Regime.charging.name), initial_input_time_seq)[
            1].real  # [:len(t)]
        # convolve(i1_charging, self.ts_and_impulse_response_charging[1][:, 0, 0]).real

        o11 = o11_from_convolve_charging
        while t[-1] < tend + Dt_MESS:
            i1_discharging = numpy.piecewise(t, [t > t_discharging_start, ],
                                             [lambda t: numpy.interp(t - Dt_MESS, _get_ts(o11), o11), 0])
            # o11_from_convolve_discharging =  convolve(
            #     i1_discharging,
            #     self.ts_and_impulse_response_discharging[1][:,0,0], ).real
            o11_from_convolve_discharging = self.calculate_response_time_seq_ij(
                (self.impulse_response_time_seq_discharging[0], self.impulse_response_time_seq_discharging[1][:, 0, 0]),
                (t, i1_discharging))[1].real
            n = len(o11_from_convolve_discharging) - len(o11_from_convolve_charging)
            if n >= 0:
                o11 = self._add_signals_with_the_same_start_time(o11_from_convolve_charging,
                                                                 o11_from_convolve_discharging)
                # logger.info("len(o33_from_convolve_discharging) - len(o33_from_convolve_charging) > 0")
            else:
                raise RuntimeError(
                    "It should be len(o33_from_convolve_discharging) - len(o33_from_convolve_charging) >= 0")
                # o11 = o11_from_convolve_charging[:len(o11_from_convolve_discharging)] + o11_from_convolve_discharging
                # logger.info("else")
            # if  _get_ts(i3_discharging,) > tend :break
            t = numpy.arange(t[0], t[-1] + Dt_MESS, self.dt)

        o21_charging = self.calculate_response_time_seq_ij(
            self.get_impulse_response_time_seq_ij(1, 0, self.Regime.charging.name), initial_input_time_seq)[1].real
        # convolve(i1_charging, self.ts_and_impulse_response_charging[1][:, 1, 0])
        # o21_discharging =  convolve(i1_discharging, self.ts_and_impulse_response_discharging[1][:, 1, 0])

        o21_discharging = self.calculate_response_time_seq_ij(
            self.get_impulse_response_time_seq_ij(1, 0, self.Regime.discharging.name),
            (_get_ts(i1_discharging), i1_discharging))[1].real
        o21 = self._add_signals_with_the_same_start_time(o21_charging, o21_discharging).real
        trusted_df = lambda df: df[df[0] <= tend]
        df_i1_charging = trusted_df(to_df(numpy.vstack((_get_ts(i1_charging), i1_charging)).T, 2 / self.f_ref))
        df_i1_discharging = trusted_df(to_df(numpy.vstack((_get_ts(i1_discharging), i1_discharging)).T, 2 / self.f_ref))
        df_o11 = trusted_df(to_df(numpy.vstack((_get_ts(o11, ), o11)).T, 2 / self.f_ref))
        df_o21 = trusted_df(to_df(numpy.vstack((_get_ts(o21, ), o21)).T, 2 / self.f_ref))

        return df_i1_charging, df_i1_discharging, df_o11, df_o21


def get_S_parameter_from_CST_proj(cst_proj: cst.results.ProjectFile, S_parameter_name: str, run_id=0):
    """

    :param cst_proj:
    :param S_parameter_name: such as "S3,3"
    :return:
    """
    return numpy.array(
        cst_proj.get_3d().get_result_item('1D Results\\S-Parameters\\%s' % S_parameter_name, run_id).get_data())


def build_network_from_CST_S_data(S11: numpy.ndarray, S21: numpy.ndarray, *args, **kwargs):
    return skrf.Network(frequency=skrf.Frequency.from_f(S11[:, 0].real, unit="GHz"),
                        s=numpy.array([[S11[:, 1], S21[:, 1], ],
                                       [S21[:, 1], S11[:, 1]]]).transpose((2, 0, 1)),
                        # name = "Network_charging"
                        #        z0 =numpy.array( [ S33_charging[:, 2], 50*numpy.ones(( S33_charging[:, 2].shape)), ]).T
                        *args, **kwargs).extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))


def build_network_of_waveguide(gamma_data_from_CST: numpy.ndarray, resampled_freq_unit_in_GHz=None,
                               L_tranline_port3=1.0, ignore_attenuation=False
                               ):
    gamma_3_interpolator = interp1d(gamma_data_from_CST[:, 0].real, gamma_data_from_CST[:, 1], fill_value=0j,
                                    bounds_error=False)
    if resampled_freq_unit_in_GHz is None:
        resampled_freq_unit_in_GHz = gamma_data_from_CST[:, 0].real  # compressor.nw_charging.f

    S_tranline_port3 = numpy.zeros((len(resampled_freq_unit_in_GHz), 2, 2), dtype=complex)
    # L_tranline_port3 =-( 300-40-20)*1e-3
    S_tranline_port3[:, 0, 1] = S_tranline_port3[:, 1, 0] = numpy.exp(
        -gamma_3_interpolator(resampled_freq_unit_in_GHz) * L_tranline_port3)
    return skrf.Network(frequency=skrf.Frequency.from_f(resampled_freq_unit_in_GHz, unit="GHz"), s=S_tranline_port3,
                        # z0=numpy.interp(resamped_f,S33_charging[:,0].real,S33_charging[:,2],)
                        ).extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))


if __name__ == '__main__':
    cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch_1-1.paramsweep.cst"
    # cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch.TapperedSwitchCav.2.paramsweep.cst"
    # cst_proj_path =         r"E:\CSTprojects\rfCompressor\cascadedHT\SES_allCST.cst"
    proj_discharging: cst.results.ProjectFile = cst.results.ProjectFile(cst_proj_path,
                                                                        allow_interactive=True)
    proj_charging: cst.results.ProjectFile = cst.results.ProjectFile(
        proj_discharging.filename[:
                                  -len("paramsweep.cst")
        # -len("cst")
        ]
        + "ES.cst",
        allow_interactive=True)

    run_id = 0
    run_id_charging = 0

    gamma_3 = numpy.array(
        proj_charging.get_3d().get_result_item('1D Results\\Port Information\\Gamma\\3(1)', run_id_charging).get_data())
    # gamma_3[:,1] = gamma_3[:,1] .imag*1j# Ignoring attenuation

    S33_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S3,3", run_id)
    S23_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S2,3", run_id)

    S33_charging = get_S_parameter_from_CST_proj(proj_charging, "S3,3", run_id)
    S23_charging = get_S_parameter_from_CST_proj(proj_charging, "S2,3", run_id)
    dt = 1 / f_ref / 5.  # 1/f_target / 10
    ts = numpy.arange(-20e-9, 0, dt)
    # ts = ts[:(len(ts)//2)*2]
    # resampled_f = numpy.arange(9e9, 9.6e9,0.0006000000000000015e9)
    resampled_f = numpy.arange(9e9, 9.6e9,0.1e9)
    # resampled_f = numpy.arange(0e9, 12e9,0.1e9)
    # resampled_f = numpy.arange(9e9, 9.6e9,0.0006100000000000015e9)
    # compressor.nw_charging = compressor.nw_charging.interpolate(resampled_f)#
    # compressor.nw_discharging = compressor.nw_discharging.interpolate(resampled_f)#[band]
    compressor = Compressor(build_network_from_CST_S_data(S33_discharging, S23_discharging).interpolate(resampled_f).extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))
                         ,   build_network_from_CST_S_data(S33_charging, S23_charging).interpolate(resampled_f).extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2))),
                            )

    ts_for_caching_impulse_response =numpy.arange(-400e-9, 0, dt)



    compressor_deembedded = compressor.embed_with_waveguide(gamma_3, -((300 - 40 - 20)) * 1e-3)

    # band = "9-9.6GHz"
    # resampled_f = numpy.arange(9e9, 9.6e9,0.0006000000000000015e9)
    # compressor.nw_charging = compressor.nw_charging.interpolate(resampled_f)#
    # compressor.nw_discharging = compressor.nw_discharging.interpolate(resampled_f)#[band]
    # compressor_deembedded.nw_charging =compressor_deembedded.nw_charging.interpolate(resampled_f)#[band]
    # compressor_deembedded.nw_discharging = compressor_deembedded.nw_discharging.interpolate(resampled_f)#[band]
    # compressor.cache_impulse_responses(ts_for_caching_impulse_response)
    # compressor_deembedded.cache_impulse_responses(ts_for_caching_impulse_response)

    sin = numpy.sin(2 * numpy.pi * f_ref * ts)

    df_i3_charging, df_i3_discharging, df_o33, df_o23 = compressor.run((ts, sin), 10e-9,
                                                                       compressor.correct_Dt_MESS(
                                                                           f_ref, 15e-9))
    plt.figure()
    plt.plot(df_o23[0],df_o23[1])

    plt.figure()

    for Dt_MESS in numpy.array((*numpy.linspace(0., 1e-9, 5),
                                 *numpy.linspace(1e-9, 15e-9, 5),
                                       # *numpy.linspace(14e-9, 15e-9, 5),
                                )

    ):
        fixed_Dt_MESS =    compressor_deembedded.correct_Dt_MESS(
            f_ref, Dt_MESS)
        df_i3_charging, df_i3_discharging, df_o33, df_o23 = compressor_deembedded.run((ts, sin), 50e-9,
                                                                                   fixed_Dt_MESS, )
        plt.plot(df_o23[0] / 1e-9, df_o23[key_complex].abs() ** 2, label="Dt_MESS = %.2f ns" % (fixed_Dt_MESS / 1e-9))

    plt.legend()

    plt.figure()
    convolved =compressor.calculate_response_time_seq_ij(
        compressor.get_impulse_response_time_seq_ij(0, 0,Compressor.Regime.charging.name),
        (ts,numpy.sin(2*numpy.pi*f_ref * ts)))
    plt.plot(convolved[0] / 1e-9, convolved[1])

    plt.figure()
    compressor.nw_charging.plot_s_mag()

    plt.figure()
    ir = compressor.nw_charging.impulse_response()
    plt.plot(ir[0], ir[1][:, 0, 0])
    convolved  = convolve(ir[1][:, 0, 0], numpy.sin(2*numpy.pi*f_ref * ir[0]))[len(ir[0])//2:]
    plt.figure()
    plt.plot(extend_time_sequence(ir[0],convolved),convolved)
    plt.figure()
    plt.plot(ir[0], numpy.sin(2*numpy.pi*f_ref * ir[0]))
