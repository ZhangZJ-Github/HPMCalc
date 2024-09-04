# -*- coding: utf-8 -*-
# @Time    : 2024/7/8 21:44
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : export_results.py
# @Software: PyCharm
import json
import os
import typing

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import shapely.geometry
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from _logging import logger
from time_dependent_output_2 import _func_eta_OM_peak

matplotlib.use('tkagg')
import matplotlib.colors as mcolors

import matplotlib.colorbar
import skrf

import matplotlib.pyplot as plt
from time_dependent_output_2 import OM_2port, S_param_network, Compressor
import scipy.constants as C
plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'

key_signals = 'signal'
key_Sparameters = 'S'
key_parameter_combination = 'parameter_combination'
key_result_dir = 'result_dir'

key_time = 'time/ns'
key_power_balance = 'Balance [3]'  # 定义同CST
key_freq = 'freq/GHz'
key_S23 = 'S2,3'
key_S33 = 'S3,3'
key_Zref3  = 'Zref3/Ohm'
f_target = 9.3e9
df_to_plots=  {}
datas = {}
def get_network_from_CST_S_data(S33_charging:numpy.ndarray,S23_charging:numpy.ndarray):
    return skrf.Network(frequency=skrf.Frequency.from_f(S33_charging[:, 0].real, unit="GHz"),
                 s=numpy.array([[S33_charging[:, 1], S23_charging[:, 1], ],
                                [S23_charging[:, 1], S33_charging[:, 1]]]).transpose((2, 0, 1)),
                 # name = "Network_charging"
                 #        z0 =numpy.array( [ S33_charging[:, 2], 50*numpy.ones(( S33_charging[:, 2].shape)), ]).T
                 )

def load_data(path) -> dict:
    data = {}

    for run_id_dir in os.listdir(path):
        dir = os.path.join(path, run_id_dir)
        # logger.info(dir)

        df_S = pandas.read_csv(dir + '/S-parameters.csv'
                               )
        df_S_complex_columns = list(df_S.columns)
        df_S_complex_columns.remove(key_freq)
        df_S[df_S_complex_columns] = df_S[df_S_complex_columns].astype(complex)
        signal_dir = dir + '/signals'
        signals = {}
        for signal_csv_name in os.listdir(signal_dir):
            signal_name = signal_csv_name[:-len('.csv')]
            signals[signal_name] = pandas.read_csv(r'%s/%s' % (signal_dir, signal_csv_name))
        with open(r'%s/parameter_combination.json' % dir, 'r') as f:
            parameter_combination = json.load(f)
        data[int(run_id_dir)] = {key_Sparameters: df_S, key_signals: signals,
                                 key_parameter_combination: parameter_combination, key_result_dir: dir}
        logger.info(dir)
    return data


# aaa
# data = load_data(r"F:\changeworld\HPMCalc\theory\rfCompressor\test_SES_switch01.1.cst.results")
# aaaa

cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch_1-1.paramsweep.cst"
# cst_proj_path = r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch01.1.cst"
proj: cst.results.ProjectFile = cst.results.ProjectFile(cst_proj_path,
                                                        allow_interactive=True)
proj_charging: cst.results.ProjectFile = cst.results.ProjectFile(
    cst_proj_path[:-len("paramsweep.cst")] + r"ES.cst",
    allow_interactive=True)
default_runid = 1
S33_charging = numpy.array(
    proj_charging.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3',default_runid ).get_data())
S23_charging = numpy.array(
    proj_charging.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3', default_runid).get_data())
gamma_3 =numpy.array( proj_charging.get_3d().get_result_item('1D Results\\Port Information\\Gamma\\3(1)',default_runid).get_data())
gamma_2 =numpy.array( proj_charging.get_3d().get_result_item('1D Results\\Port Information\\Gamma\\2(1)',default_runid).get_data())

Z_ref3 =   numpy.array((proj.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 3(1)',default_runid)).get_data())
unique_str = os.path.split(cst_proj_path)[1]

# proj_discharging :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch02.cst",
#                                                         allow_interactive=True)
# proj_3D_charging :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch02.1.cst",
#                                                         allow_interactive=True)
run_ids = proj.get_3d().get_all_run_ids()
data = {}


def signal_to_df(o23: numpy.ndarray):
    df_o23 = pandas.DataFrame(o23, columns=[key_time, key_signals])
    df_o23[key_period_for_calculate_avg] = df_o23[key_time] * 1e-9 // (2 / f_target)
    df_o23[key_square] = df_o23[key_signals] ** 2
    df_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
    df_o23[key_interpolated_periodic_avg_square] = numpy.interp(df_o23[key_time], df_time_avg[key_time],
                                                                df_time_avg[key_square], )
    df_o23[key_complex] = scipy.signal.hilbert(df_o23[key_signals])
    return df_o23


def get_signal_column_name(signalname: str):
    return "%s, %s" % (signalname, key_time), signalname
run_ids:list
run_ids.remove(0)
for run_id in run_ids:
    try:
        S33_3D = numpy.array(proj.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3', run_id=run_id, ).get_data())
        S23_3D = numpy.array(proj.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3', run_id=run_id, ).get_data())
        # S22_3D = numpy.array(proj_discharging.get_3d().get_result_item('1D Results\\S-Parameters\\S2,2', run_id=run_id, ).get_data())
        # S32_3D = numpy.array(proj_discharging.get_3d().get_result_item('1D Results\\S-Parameters\\S3,2', run_id=run_id, ).get_data())
        # power_balance =(numpy.abs(S23_3D[:,1])**2+numpy.abs(S33_3D[:, 1])**2)
        # sig_i3 = numpy.array(
        #     proj_discharging.get_schematic().get_result_item('Tasks\\Tran1\\TD Signals\\I3', run_id=run_id, ).get_data())
        # sig_o23 = numpy.array(
        #     proj_discharging.get_schematic().get_result_item('Tasks\\Tran1\\TD Signals\\O2,3', run_id=run_id, ).get_data())
        # sig_o33 = numpy.array(
        #     proj_discharging.get_schematic().get_result_item('Tasks\\Tran1\\TD Signals\\O3,3', run_id=run_id, ).get_data())

        df_S = pandas.DataFrame(data={
            key_freq: numpy.real(S23_3D[:, 0]),
            # 'S2,2': (S22_3D[:, 1]),
            key_S23: (S23_3D[:, 1]),
            # 'S3,2': (S32_3D[:, 1]),
            key_S33: (S33_3D[:, 1]),
            # 'Zref2/Ohm': S22_3D[:, 2],
            key_Zref3: S33_3D[:, 2],
            # 'Balance [3]': power_balance
        })
        # signals = {'i3': signal_to_df(sig_i3),
        #            'o23': signal_to_df(sig_o23),
        #            'o33': signal_to_df(sig_o33),
        #            }
        result_dir = '%s.results/%04d' % (os.path.split(cst_proj_path)[1], run_id)
        os.makedirs(result_dir, exist_ok=True)
        signals_dir = '%s/signals' % result_dir
        os.makedirs(signals_dir, exist_ok=True)

        df_S.to_csv(r'%s/S-parameters.csv' % result_dir, index=False)
        # for signal_name in signals:
        #     signals[signal_name].to_csv(r'%s/%s.csv' % (signals_dir, signal_name), index=False)
        parameter_combination = proj.get_3d().get_parameter_combination(run_id)
        with open(r'%s/parameter_combination.json' % result_dir, 'w') as f:
            json.dump(parameter_combination, f)
        data[run_id] = {key_Sparameters: df_S,  # key_signals: signals,
                        key_parameter_combination: parameter_combination, key_result_dir: (result_dir)}
        # logger.info(r'"%s" done' % (os.path.abspath('%s/parameter_combination.json' % result_dir)))
    except ValueError as e:logger.warning(e)

datas[cst_proj_path]=data
key_zoff = 'SwitchZoffset'
key_yoff = 'SwitchRoffset'
key_normalized_Pout_inf = 'normalized_Pout_inf'
key_runid = 'run_id'
key_signal_fft_at_f_target = "signal_fft_at_f_target"
key_I_pe = 'I_pe'
key_I_pc = 'I_pc'
key_Dt_OM = "Dt_OM"
key_eta_max_of_core_structure ="eta_max_of_core_structure"
key_G_cav_of_core_structure = "G_cav_of_core_structure"
key_G_of_core_structure = "G_of_core_structure"

# resampled_time = numpy.arange(-50e-9, 50e-9, 0.01e-9)
# resampled_dt = numpy.diff(resampled_time)[0]  # ns
# freqs = scipy.fft.fftfreq(len(resampled_time), resampled_dt)  # unit in Hz

def fix_network(nw:skrf.Network
        #nw_charging_core_s:numpy.ndarray

                    ):
    freq = nw.f
    s = nw.s

    # nw_charging_core_s = nw_charging_core.s
    # nw_charging_core_s = nw_charging_core_s_
    s[freq  < 6.0e9, :, :] = 0.
    indx = (numpy.abs(s) > 1)
    where = numpy.where(indx)
    # logger.info(where)
    # s[indx] = numpy.nan#1.*numpy.exp(1j*numpy.angle(s[indx]))
    freq[where[0]] = numpy.nan

    return skrf.Network(frequency=freq[~numpy.isnan( freq)],s = s[~numpy.isnan( freq)],)#,where






z_port3 = 20e-3
z_short_terminal = 299.72e-3
z_port2_centerline =(z_short_terminal*1e3-proj.get_3d().get_parameter_combination(run_id)['Dz2'] )*1e-3

L_storage_cav_simulated = z_short_terminal - z_port3
impulse_duration = 100.0e-9
impulse_dt = 1 / (f_target) / 20
resamped_f = S23_3D[:,0].real
S_tranline_port3 = numpy.zeros((len(resamped_f), 2, 2), dtype = complex)
L_tranline_port3 =-(z_port2_centerline  -       (15) *1e-3 -z_port3)
L_OM = L_storage_cav_simulated +L_tranline_port3
gamma_3_interpolated = numpy.interp(resamped_f,gamma_3[:,0].real,gamma_3[:, 1], )
v_p = 2 * numpy.pi * f_target / numpy.interp((f_target / 1e9), gamma_3[:, 0].real, gamma_3[:, 1]).imag
v_g = C.c **2/ v_p
S_tranline_port3[:, 1, 0] = numpy.exp(-gamma_3_interpolated* L_tranline_port3)
# S_tranline_port3[:, 1, 0][  S23_3D[:,0 ]<6.0 ] = complex(0)
S_tranline_port3[:, 0, 1]= S_tranline_port3[:, 1, 0]
slicing = "0-12GHz"
# tranline_port3_nw = skrf.Network(r"E:\CSTprojects\rfCompressor\cascadedHT\Rectangular_wg.s2p")
tranline_port3_nw = skrf.Network(frequency =skrf.Frequency.from_f (resamped_f,unit = "GHz"), s =S_tranline_port3,
                                 # z0=numpy.interp(resamped_f,S33_charging[:,0].real,S33_charging[:,2],)
                                 )[slicing]

# tranline_port3_nw.resample(S23_3D[:,0],unit= 'GHz')
# plt.figure()
# plt.plot(tranline_port3_nw.f ,numpy.abs(tranline_port3_nw.s[:, 0,1]))
def cascade_WG_and_nw( nw_WG:skrf.Network, nw:skrf.Network)->skrf.Network:
    f = nw.f
    s = 0* nw.s

    nw_WG_resampled = nw_WG.interpolate(f)
    s[:,1,1] =s[:, 0,0] = nw_WG_resampled.s[:, 0, 0] + nw.s[:, 0, 0] * nw_WG_resampled.s[:, 0, 1] * nw_WG_resampled.s[:, 1, 0]
    s[:,0, 1]  = s[:, 1,0] = nw_WG_resampled.s[:, 1, 0] * nw.s[:, 1, 0]

    return skrf.Network(frequency=f, s = s)


# from  time_dependent_output_2 import S_param_network,OM_2port

nw_charging = get_network_from_CST_S_data(S33_charging, S23_charging)[slicing]
s_=  nw_charging.interpolate([f_target]).s[0]


df_to_plot = pandas.DataFrame(
    columns=[key_runid, key_zoff, key_yoff, 'tau',key_Dt_OM, key_normalized_Pout_inf, key_signal_fft_at_f_target,
             key_power_balance, key_S23, key_result_dir, key_I_pe, key_I_pc,
             key_eta_max_of_core_structure,key_G_cav_of_core_structure,key_G_of_core_structure
             ])
for run_id in data:
    eta_inf = 1
    tau = 1e-9
    S23_, S33_ = interp1d(data[run_id][key_Sparameters][key_freq].values, numpy.vstack(
        [data[run_id][key_Sparameters][key_S23], data[run_id][key_Sparameters][key_S33]]))(f_target / 1e9)

    Dt_right = 1e-9
    eta_max_of_core_structure = 0.





    # sig = nw_discharging_core.impulse_response()
    # # plt.figure()
    # plt.plot(sig[0], sig[1][:, 1, 0], '.-')
    #


    if True:
        nw_discharging = get_network_from_CST_S_data(data[run_id][key_Sparameters][[key_freq, key_S33,key_Zref3]].values,
                                                     data[run_id][key_Sparameters][[key_freq, key_S23]].values)[slicing]
        # De-embedding
        # nw_charging_core =tranline_port3_nw.inv**nw_charging
        # nw_discharging_core =  tranline_port3_nw.inv ** nw_discharging

        # plt.figure()
        # plt.plot(numpy.abs(nw_discharging_core.s[:, 1, 0]))
        # plt.plot(numpy.abs(nw_discharging.s[:, 1, 0]))
        nw_charging_core = cascade_WG_and_nw(tranline_port3_nw,nw_charging)
        nw_discharging_core = cascade_WG_and_nw(tranline_port3_nw,nw_discharging)
        nw_charging_core=fix_network(nw_charging_core )
        nw_discharging_core=fix_network(nw_discharging_core )


        # nw_charging_core.resample(resamped_f)
        # nw_discharging_core.resample(resamped_f)
        compressor = Compressor(
            OM_2port(
                S_param_network(numpy.vstack((nw_charging_core.f,nw_charging_core.s[:,0,0])).T, impulse_duration, impulse_dt),
                S_param_network( numpy.vstack((nw_charging_core.f, nw_charging_core.s[:, 1, 0])).T, impulse_duration, impulse_dt)),
            OM_2port(
                S_param_network(numpy.vstack((nw_discharging_core.f,nw_discharging_core.s[:,0,0])).T, impulse_duration,
                                impulse_dt),
                S_param_network(numpy.vstack((nw_discharging_core.f,nw_discharging_core.s[:,1,0])).T, impulse_duration,
                                impulse_dt)), )

        # compressor = Compressor(
        #     OM_2port(
        #         S_param_network(numpy.vstack((nw_charging_core.f,nw_charging_core.s[:,0,0])).T, impulse_duration, impulse_dt),
        #         S_param_network( numpy.vstack((nw_charging_core.f, nw_charging_core.s[:, 1, 0])).T, impulse_duration, impulse_dt)),
        #     OM_2port(
        #         S_param_network(numpy.vstack((nw_discharging_core.f,nw_discharging_core.s[:,0,0])).T, impulse_duration,
        #                         impulse_dt),
        #         S_param_network(numpy.vstack((nw_discharging_core.f,nw_discharging_core.s[:,1,0])).T, impulse_duration,
        #                         impulse_dt)), )
        # t_ = numpy.arange(0 , 40e-9,compressor.dt)
        # sin_ = numpy.sin(2*numpy.pi *f_target*t_)
        # import time_dependent_output_2
        # plt.plot(time_dependent_output_2.get_ts(compressor.om_discharging.S21.impulse_response,0,compressor.dt),compressor.om_discharging.S21.impulse_response)
        #
        # plt.plot(t_, scipy.signal.convolve(compressor.om_discharging.S21.impulse_response,sin_)[:len(t_)])

        Dt_MESS = compressor.correct_Dt_MESS(f_target, impulse_duration*0+0.e-9)
        L_MESS = v_g *Dt_MESS/2
        G_cav_of_core_structure = 1 / ((1 - numpy.abs(s_[0, 0]) ** 2) / L_storage_cav_simulated * (L_OM+L_MESS))

        t_charging_start = -2 * max(Dt_MESS, impulse_duration)
        t_discharging_start = 0.0
        tend = 20e-9 + Dt_MESS#100.0e-9
        df_i3_charging, df_i3_discharging, df_o33, df_o23 = compressor.run(f_target, t_charging_start,
                                                                           t_discharging_start,
                                                                           tend, Dt_MESS)

        trusted_index1 = lambda df_o23: (df_o23[0] < tend)
        trusted_index2 = lambda df_o23: (df_o23[0] < tend) &(df_o23[0] >t_discharging_start)

        eta_max_of_core_structure = max(  ((df_o23[trusted_index2(df_o23)][key_interpolated_periodic_avg_square]) * 2) )

        # plt.figure()
        # plt.plot(df_o23[trusted_index1(df_o23)][0],
        #          ((df_o23[trusted_index1(df_o23)][key_interpolated_periodic_avg_square]) * 2) ** 0.5,
        #          label="%.2f ns, L_OM = %.2f mm, L_MESS = %.2f mm" % (Dt_MESS * 1e9,L_OM*1e3, L_MESS  /1e-3))
        # plt.plot(df_i3_discharging[trusted_index1(df_i3_discharging)][0],
        #          ((df_i3_discharging[trusted_index1(df_i3_discharging)][
        #              key_interpolated_periodic_avg_square]) * 2) ** 0.5)
        # # plt.plot(df_o23[trusted_index1][0], ((df_o23[trusted_index1][1]) ) )
        # plt.legend()


        #
        # aaaaa
        # trusted_index = (df_o23[0] > t_discharging_start) & (df_o23[0] < Dt_MESS)
        #
        # eta_inf = numpy.abs(S23_) ** 2  # (df_o23[trusted_index][key_interpolated_periodic_avg_square].values[-2] * 2)
        # __func_eta_OM_peak = lambda Dt_MESS, tau, Dt_right: _func_eta_OM_peak(Dt_MESS, eta_inf, tau, Dt_right)
        # (eta_inf, tau, Dt_right), cov = curve_fit(_func_eta_OM_peak,
        #                                           df_o23[trusted_index][0].values,
        #                                           df_o23[trusted_index][
        #                                               key_interpolated_periodic_avg_square].values * 2,
        #                                           p0=[eta_inf, 1e-9, -1e-9],
        #                                           bounds=numpy.array(((0, 1), (0, 100e-9,), (-100e-9, 0))).T)
        # # S  = numpy.zeros(S33_charging.shape)
        # logger.info(numpy.array([eta_inf, tau, Dt_right]))
    logger.info("run_id=%d"%run_id)
    df_to_plot.loc[len(df_to_plot)] = {
        key_runid: run_id,
        key_zoff: data[run_id][key_parameter_combination][key_zoff],
        key_yoff: data[run_id][key_parameter_combination][key_yoff],
        'tau': tau,
        key_Dt_OM: Dt_right,
        key_normalized_Pout_inf: eta_inf,
        # key_signal_fft_at_f_target: signal_fft_at_f_target,
        key_power_balance: (numpy.abs(S23_) ** 2 + numpy.abs(S33_) ** 2) ** 0.5,
        key_S23: numpy.abs(S23_),
        key_result_dir: data[run_id][key_result_dir],
        key_eta_max_of_core_structure: eta_max_of_core_structure,
        key_G_cav_of_core_structure:G_cav_of_core_structure,
        key_G_of_core_structure: eta_max_of_core_structure*G_cav_of_core_structure,
    }

df_to_plot = df_to_plot[~pandas.isna(df_to_plot[key_runid])]

df_to_plot = pandas.concat([df_to_plot,
                            pandas.DataFrame(numpy.vstack([
                                numpy.array(
                                    numpy.meshgrid(numpy.linspace(0, 4, 5), numpy.linspace(24, 33, 5))).transpose(
                                    (1, 2, 0)).reshape((-1, 2)),
                                numpy.array(
                                    numpy.meshgrid(numpy.linspace(0, 2, 5), numpy.linspace(35, 45, 6))).transpose(
                                    (1, 2, 0)).reshape((-1, 2)),
                                numpy.array(
                                    numpy.meshgrid(numpy.linspace(26.1, 28.8, 5), numpy.linspace(35, 45, 6))).transpose(
                                    (1, 2, 0)).reshape((-1, 2)),
                                numpy.array(
                                    numpy.meshgrid(numpy.linspace(24, 28.8, 5), numpy.linspace(24.5, 33, 5))).transpose(
                                    (1, 2, 0)).reshape((-1, 2)),
                            ],
                            ), columns=[key_zoff, key_yoff]), ], axis=0, ignore_index=True)


# df_to_plot = pandas.concat([df_to_plot, pandas.read_csv("df_to_plot.FTC.csv",)],axis= 0,ignore_index=True)
# df_to_plot.loc[df_to_plot[pandas.isna(df_to_plot[key_runid])].index,['Q',key_normalized_Pout_inf]] = 1e9,0

def query(zoff, yoff, EPS=0.5,df_to_plot = df_to_plot):
    # EPS=0.5
    return df_to_plot[(numpy.abs(df_to_plot[key_zoff] - zoff) < EPS) & (numpy.abs(df_to_plot[key_yoff] - yoff) < EPS)]


# plt.figure()
# plt.scatter(*df_to_plot[[key_zoff, key_yoff]].values.T)


# df_seqs = [df_seq1, df_seq2, df_seq3]


def plot_contour(df_seq: pandas.DataFrame, how_to_process_df_seq, ax: plt.Axes, **kwargs):
    Z, Y = numpy.meshgrid(numpy.linspace(min(df_seq[key_zoff]), max(df_seq[key_zoff]), 100),
                          numpy.linspace(min(df_seq[key_yoff]), max(df_seq[key_yoff]), 100))
    cf = ax.contourf(Z, Y, griddata(df_seq[[key_zoff, key_yoff]].values, how_to_process_df_seq(df_seq), (Z, Y),
                                    # method='cubic'
                                    ), **kwargs)
    return cf


def queried_to_table_in_paper(df_to_plot: pandas.DataFrame):
    new_df = pandas.DataFrame()
    new_df["case"] = ["A%d" % (i + 1) for i in df_to_plot.index]
    new_df["zoff"] = df_to_plot[key_zoff]
    new_df["yoff"] = df_to_plot[key_yoff]
    new_df["S21"] = df_to_plot[key_S23]
    new_df["1/(1-sum)"] = 1 / (1 - df_to_plot[key_power_balance] ** 2)
    new_df["Ipe"] = df_to_plot[key_I_pe]
    new_df["Ipc"] = df_to_plot[key_I_pc]
    return new_df
# __selected_pts = {'E:\\CSTprojects\\rfCompressor\\cascadedHT\\SES_switch_2-1.paramsweep.cst':
#                 (    numpy.array([[9.5, 37.6],
#                                  [19.6, 31.6]]),
#                      r"$0.5\lambda_g$::"
#                      )}


def get_selected_pts(cst_proj_path ):
    return {'E:\\CSTprojects\\rfCompressor\\cascadedHT\\SES_switch_2-1.paramsweep.cst':
        numpy.array([  # [9.5, 37.6],
            # [19.6, 31.6],
            [8.75, 37],
            [20, 31.5]
        ]
        ),
        'E:\\CSTprojects\\rfCompressor\\cascadedHT\\SES_switch_1-1.paramsweep.cst': numpy.array([
            # [9,30],
            [8.6, 30.7],
            [11.375, 28.225],
            [19.2, 32],
            # [22.5,31.0]
        ]
        ),
        'E:\\CSTprojects\\rfCompressor\\cascadedHT\\SES_switch_1-1.half_cav.paramsweep.cst': numpy.array([
            # [9,30],
            # [7.44, 25],
            # [22.5,31.0]
        [8.56,30.5],
        ]
        ),
        'E:\\CSTprojects\\rfCompressor\\cascadedHT\\SES_switch_3-1.paramsweep.cst': numpy.array([
            # [9,30],
            [12.857143, 35.25],
            # [22.5,31.0]
        ]
        ),
    }.get(cst_proj_path, numpy.array([[]]))
selected_pts =  get_selected_pts(cst_proj_path)
# selected_pts = numpy.array([
#     [2.997, 15.75],
#     [10.8, 22.8],
#     [15.9, 26.4],
#     # [17.97,28.92],
#     [18.8, 30.0],
#     # [19.8,32.2],
#     # [19.3,35.7],
#     # [10.5, 38.8],
#     # [20.4, 27.6],
#     # [21.0, 41.8],
# ])


# df_to_plot[key_I_pe] = df_to_plot[key_S23] ** 2 * df_to_plot[key_power_balance] ** 4 / (1 - df_to_plot[key_power_balance] ** 2) ** 2
df_to_plot[key_I_pe] =( df_to_plot[key_S23]/ (df_to_plot['tau']+0.1e-9)) ** 2
key_response_speed = "response_speed (1/(1-PB^2))"
df_to_plot[key_response_speed] = 1 / (1 - df_to_plot[key_power_balance] ** 2)


Z, Y = numpy.meshgrid(numpy.linspace(244, 305, 100), numpy.linspace(-17, 37, 50))
df_ptlist = pandas.DataFrame({'x': numpy.zeros(Z.ravel().shape) + 1, 'y': Y.ravel(), 'z': Z.ravel(), })
df_ptlist.to_csv(os.path.join(os.path.split(proj_charging.filename)[0], r"ptlist.txt"),
                 index=False, sep=' ', header=None
                 )

df_Eabs = pandas.read_csv(
    os.path.join(os.path.splitext(proj_charging.filename)[0], r'Export\3d\e-field (f=f0) [3].txt'),
    sep=r'\s\s+',
    skiprows=lambda
        idx: idx == 1)  # pandas.read_csv(r"E:\CSTprojects\rfCompressor\Eabs.txt",sep=r'\s\s+',skiprows=lambda idx:idx==1)
# Z,Y = numpy.meshgrid(numpy.linspace(min(df_Eabs["z [mm]"]),max(df_Eabs["z [mm]"]),1000),numpy.linspace(min(df_Eabs["y [mm]"]),max(df_Eabs["y [mm]"]),500))


# plt.scatter(*df_Eabs[["z [mm]","y [mm]",]].values.T)


colnames_E = []
for col in df_Eabs.columns:
    if col.startswith("E"):
        colnames_E.append(col)
Eabs = (df_Eabs[colnames_E] ** 2).sum(axis=1) ** 0.5
Eabs_max = Eabs.max()
df_to_plot[key_I_pc] = numpy.nan
key_Eabs_max = 'Eabs_max'
df_to_plot[key_Eabs_max] = numpy.nan
key_Eabs_max_tube = 'Eabs_max_tube'
df_to_plot[key_Eabs_max_tube] = numpy.nan
# zoff0, yoff0 = 266.71, -13.4
zoff0, yoff0 = z_short_terminal*1e3 - proj_charging.get_3d().get_parameter_combination(0)["Dz3"] - 28 / 2, -13.4

r_tube = 3
pts_to_calc_tube_max = numpy.array(
    (numpy.meshgrid(numpy.linspace(-r_tube, r_tube, 50), numpy.linspace(-r_tube, r_tube, 50)))).transpose(
    (1, 2, 0)).reshape((-1, 2))
pts_to_calc_tube_max = pts_to_calc_tube_max[
    pts_to_calc_tube_max[:, 0] ** 2 + pts_to_calc_tube_max[:, 1] ** 2 <= r_tube ** 2]
# plt.figure()
# plt.scatter( *pts_to_calc_tube_max.T)


from scipy.interpolate import LinearNDInterpolator

Eabs_interpolator = LinearNDInterpolator(df_Eabs[["z [mm]", "y [mm]", ]].values, Eabs)
for i in df_to_plot[~pandas.isna(df_to_plot[key_runid])].index:
    zoff, yoff = df_to_plot[[key_zoff, key_yoff]].loc[i]
    Eabs_max_tube = (Eabs_interpolator(numpy.array([zoff0 + zoff, yoff0 + yoff]) + pts_to_calc_tube_max)).max()
    df_to_plot.loc[i, key_Eabs_max_tube] = Eabs_max_tube
    df_to_plot.loc[i, key_I_pc] = (Eabs_max / Eabs_max_tube) ** 2
df_to_plot[key_Eabs_max] = Eabs_max

querieds = pandas.DataFrame(columns=df_to_plot.columns)

for i, pt in enumerate(selected_pts):
    if len(pt):    querieds.loc[len(querieds)] = query(*selected_pts[i, :], 0.001).iloc[0]

df_to_plot.to_csv("export_result2.df_to_plot.%s.csv"%unique_str,index= False)
df_to_plots [cst_proj_path] = df_to_plot







queried_to_table_in_paper(querieds).to_csv("temp.csv")
# fig: plt.Figure = plt.figure(constrained_layout=True, figsize=(5.7, 6.7))
# gs = plt.GridSpec(3, 2, figure=fig, )
# axs = [fig.add_subplot(gs_, ) for gs_ in (gs[:2, :], gs[2,0], gs[2, 1])]
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, constrained_layout=True, figsize=(3.8, 8))

for ax in axs[1:]:
    ax.sharex(axs[0])
    ax.sharey(axs[0])
df_seqs = [df_to_plot]
for df_seq in df_seqs:
    f1 = lambda df_seq:df_seq[key_G_of_core_structure]# df_seq[key_I_pe]
    # levels1 = numpy.linspace(0, 7200, 20)
    cf1 = plot_contour(df_seq, f1, axs[0], levels=20#levels1
                       )
    # f1 = lambda df_seq: df_seq[key_I_pe]
    # cf2 = plot_contour(df_seq, f1, axs[1], levels= levels1)
    f1 = lambda df_seq: df_seq[key_S23]
    cf3 = plot_contour(df_seq, f1, axs[1], levels=numpy.linspace(0, 1, 20)
                       # numpy.linspace(min(f1(df_to_plot)), max(f1(df_to_plot)), 100)
                       )
    f1 = lambda df_seq: df_seq[key_eta_max_of_core_structure] #1 / (1 - df_to_plot[key_power_balance] ** 2)  # df_seq["tau"]
    cf4 = plot_contour(df_seq, f1, axs[2], levels=numpy.linspace(0, 1, 20)
                       )
# titles = ['$P_{out}(\infty) / Q$', '$P_{out}(\infty) $', '$1 / Q$', '$F(o_{23})|_{%.1fGHz} P_{out}(\infty)$'%(f_target/1e9)]
titles = ['$G$',  # '$I_{pe}$',
          '$|S_{2,1}|$',
          '$\eta_{OM,peak}$'
          ]

circ = shapely.geometry.Point(13.74, 33.13, ).buffer(5.9)
_scater_size = [0.1, 0.1, 0.1]
x_major_locator = plt.MultipleLocator(10)
y_major_locator = plt.MultipleLocator(5)

for i, ax in enumerate(axs):
    ax: plt.Axes
    ax.scatter(*df_to_plot[~pandas.isna(df_to_plot[key_runid])][[key_zoff, key_yoff]].values.T, c='k', alpha=1,
               s=_scater_size[i]
               )
    # ax.scatter(*df_to_plot[pandas.isna(df_to_plot[key_runid])][[key_zoff, key_yoff]].values.T, alpha=1, s=1)
    # ax.scatter(*numpy.array(circ.exterior.xy)[:,::4], alpha=1,s = _scater_size)
    # ax.scatter(*circ.centroid.xy, alpha=1,s = _scater_size)
    ax.set_aspect('equal')
    ax.set_title(titles[i])
    ax.set_xlabel('$z_{off}$ (mm)')
    ax.set_ylabel('$y_{off}$ (mm)')
    ax.set_ylim(25, None)
    ax.set_xlim(3, 25.)
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)

for i, cf in enumerate((cf1, cf3,cf4 )):
    # fig.colorbar(cf)
    kwargs = {}
    # if i == 1:kwargs["format"] = lambda v,pos: "%.2f"%v
    cb: matplotlib.colorbar.Colorbar = plt.colorbar(cf, ax=axs[i], **kwargs)

default_colors = list(mcolors.TABLEAU_COLORS)

for ax in axs:
    if len(querieds):ax.scatter(*querieds[[key_zoff, key_yoff]].values.T, marker='*', s=5, c='r')
for i in querieds.index:
    axs: typing.List[plt.Axes]
    # axs[0].annotate('A%d' % (i + 1), querieds.loc[i, [key_zoff, key_yoff]], (-1.5, 0.02),
    #                 textcoords='offset fontsize',
    #                 c='r')
plt.savefig('I_pe.Core_Structure' + unique_str + '.eps', )










for i in  querieds.index:
    queried = querieds.iloc[i]
    result_dir = queried[key_result_dir]
    run_id =  queried[key_runid]
    logger.info(result_dir)
    compressor = Compressor(
        OM_2port(S_param_network(S33_charging[:, :2] * (1e9, 1), impulse_duration, impulse_dt),
                 S_param_network(S23_charging[:, :2] * (1e9, 1), impulse_duration, impulse_dt)),
        OM_2port(
            S_param_network(data[run_id][key_Sparameters][[key_freq, key_S33]].values * (1e9, 1), impulse_duration,
                            impulse_dt),
            S_param_network(data[run_id][key_Sparameters][[key_freq, key_S23]].values * (1e9, 1), impulse_duration,
                            impulse_dt)), )

    ts = numpy.arange(-10e-9*0, 40e-9,compressor.dt )
    sin = numpy.sin(2*numpy.pi * f_target*ts)
    signal_o23_convolved = scipy.signal.convolve(compressor.om_discharging.S21.impulse_response, sin)
    plt.figure(figsize= (5,4),constrained_layout = True)
    df_o23 = signal_to_df(numpy.array((ts / 1e-9, signal_o23_convolved[:len(ts)].real
                                       )).T)
    trusted_index =( df_o23[key_time]>0)&( df_o23[key_time]<35)

    plt.plot(df_o23[trusted_index][key_time],( df_o23[trusted_index][key_interpolated_periodic_avg_square].values * 2)**0.5,#lw = 0.05
             label = 'simulated'
             )
    eta_inf = numpy.abs(compressor.om_discharging.S21.S_interpolator(complex(f_target)))**2

    # eta_inf = numpy.abs(S23_) ** 2  # (df_o23[trusted_index][key_interpolated_periodic_avg_square].values[-2] * 2)
    # __func_eta_OM_peak = lambda Dt_MESS, tau, Dt_right: _func_eta_OM_peak(Dt_MESS, eta_inf, tau, Dt_right)
    (eta_inf, tau, Dt_right), cov = curve_fit(_func_eta_OM_peak,
                                              df_o23[trusted_index][key_time].values*1e-9,
                                              df_o23[trusted_index][
                                                  key_interpolated_periodic_avg_square].values * 2,
                                              p0=[eta_inf, 1e-9, -1e-9],
                                              bounds=numpy.array(((0, 1), (0, 100e-9,), (-100e-9, 0))).T)
    plt.plot(ts/1e-9,_func_eta_OM_peak( ts,  eta_inf,tau, Dt_right)**0.5,'--',
             label = r"$%.2f (1-e^{\frac{t - %.2f ns}{%.2f ns}})$"%(eta_inf**0.5,  -Dt_right/1e-9, tau/1e-9))
    plt.ylim(0,1)
    plt.xlabel("time (ns)")
    plt.ylabel("normalized port voltage ($\sqrt{W}$)")
    plt.legend()
    plt.savefig("signal."+unique_str+".case%d.eps"%i)

    plt.figure(figsize= (3,3),constrained_layout = True)
    plt.plot(data[run_id][key_Sparameters][key_freq],numpy.abs(data[run_id][key_Sparameters][key_S33]),label = "$|S_{1,1}|$")
    plt.plot(data[run_id][key_Sparameters][key_freq],numpy.abs(data[run_id][key_Sparameters][key_S23]),label = "$|S_{2,1}|$")
    plt.legend()
    plt.xlim(9,9.6)
    plt.ylim(0, 1)
    plt.xlabel("frequency (GHz)")
    plt.ylabel("magnitude")
    plt.savefig("Sparam."+unique_str+".case%d.eps"%i)






# from matplotlib import colors
# norm = colors.LogNorm(vmin=Z.min(), vmax=Z.max())
plt.figure(constrained_layout=True, figsize=(3.8, 2.8))
plt.gca().set_aspect('equal')
cf = plot_contour(df_to_plot, lambda df: numpy.log10(df[key_I_pc]), plt.gca(),
                  # locator=matplotlib.ticker.LogLocator(),
                  levels=21, zorder=-1)
# cf = plot_contour(df_to_plot,lambda df: (df[key_I_pc]) ,plt.gca(),locator=matplotlib.ticker.LogLocator(numticks = 20),
#                   levels = 20,#norm = norm
#                   )
plt.gcf().colorbar(cf)
plt.scatter(*(df_to_plot[~pandas.isna(df_to_plot[key_runid])][[key_zoff, key_yoff]].values).T, c='k', alpha=1,
            s=_scater_size[1]
            )
plt.title(r"${log_{10}} {(I_{pc})}$")
ax = plt.gca()
ax.set_xlabel('$z_{off}$ / mm')
ax.set_ylabel('$y_{off}$ / mm')
ax.set_ylim(25, None)
ax.set_xlim(3, 25.)

if len(querieds):plt.scatter(*querieds[[key_zoff, key_yoff]].values.T, marker='*', s=10, c='r')
# for i in querieds.index:
#     plt.annotate('A%d' % (i + 1), querieds.loc[i, [key_zoff, key_yoff]], (0.3, 0.0),
#                  textcoords='offset fontsize',
#                  c='r')
plt.savefig("I_pc." + unique_str + ".eps", )










aaaaaaaaaa

fig,axs = plt.subplots(4,3,sharex=False,sharey=False,figsize = (6,7),constrained_layout = True)

for ii, cst_proj_path in enumerate( [
'E:\\CSTprojects\\rfCompressor\\cascadedHT\\SES_switch_1-1.paramsweep.cst',
'E:\\CSTprojects\\rfCompressor\\cascadedHT\\SES_switch_2-1.paramsweep.cst',
'E:\\CSTprojects\\rfCompressor\\cascadedHT\\SES_switch_3-1.paramsweep.cst',
]):
    querieds = pandas.DataFrame(columns=df_to_plot.columns)
    df_to_plot = df_to_plots[cst_proj_path]
    proj: cst.results.ProjectFile = cst.results.ProjectFile(cst_proj_path,
                                                            allow_interactive=True)
    proj_charging: cst.results.ProjectFile = cst.results.ProjectFile(
        cst_proj_path[:-len("paramsweep.cst")] + r"ES.cst",
        allow_interactive=True)
    df_Eabs = pandas.read_csv(
        os.path.join(os.path.splitext(proj_charging.filename)[0], r'Export\3d\e-field (f=f0) [3].txt'),
        sep=r'\s\s+',
        skiprows=lambda
            idx: idx == 1)
    selected_pts = get_selected_pts(cst_proj_path)
    for i, pt in enumerate(selected_pts):
        if len(pt):    querieds.loc[len(querieds)] = query(*selected_pts[i, :], 0.001,df_to_plot).iloc[0]



    ax = axs[0, ii]
    cf = plot_contour(df_to_plot, lambda df: numpy.log10(df[key_I_pc]), ax,
                  # locator=matplotlib.ticker.LogLocator(),
                  levels=numpy.linspace(0,4,20), zorder=-1)
    # cf = plot_contour(df_to_plot,lambda df: (df[key_I_pc]) ,plt.gca(),locator=matplotlib.ticker.LogLocator(numticks = 20),
    #                   levels = 20,#norm = norm
    #                   )
    # plt.gcf().colorbar(cf)
    # plt.scatter(*(df_to_plot[~pandas.isna(df_to_plot[key_runid])][[key_zoff, key_yoff]].values).T, c='k', alpha=1,
    #             s=_scater_size[1]
    #             )
    # plt.title(r"${log_{10}} {(I_{pc})}$")
    # ax = plt.gca()


    # for i in querieds.index:
    #     plt.annotate('A%d' % (i + 1), querieds.loc[i, [key_zoff, key_yoff]], (0.3, 0.0),
    #                  textcoords='offset fontsize',
    #                  c='r')
    # plt.savefig("I_pc." + unique_str + ".eps", )




    cf1 = plot_contour(df_to_plot, lambda df_to_plot: df_to_plot[key_S23], axs[1,ii], levels=numpy.linspace(0,1,20)#levels1
                       )
    # f1 = lambda df_seq: df_seq[key_I_pe]
    # cf2 = plot_contour(df_seq, f1, axs[1], levels= levels1)

    cf2 = plot_contour(df_to_plot, lambda df_to_plot : df_to_plot[key_eta_max_of_core_structure],axs[2,ii], levels=numpy.linspace(0, 0.5, 20)
                       # numpy.linspace(min(f1(df_to_plot)), max(f1(df_to_plot)), 100)
                       )
    cf3 = plot_contour(df_to_plot, lambda df_to_plot : df_to_plot[key_G_of_core_structure],axs[3,ii], levels=numpy.linspace(0, 400, 20)
                       # numpy.linspace(min(f1(df_to_plot)), max(f1(df_to_plot)), 100)
                       )

    # if len(querieds):
    #     for ax in axs[:,ii]:        ax.scatter(*querieds[[key_zoff, key_yoff]].values.T, marker='*', s=10, c='r')



fig:plt.Figure
fig.colorbar(cf,ax= axs[0, 2])
fig.colorbar(cf1,ax= axs[1, 2])
fig.colorbar(cf2,ax= axs[2, 2])
fig.colorbar(cf3,ax= axs[3, 2])
for ax in axs.ravel():
    ax.set_aspect('equal')

    ax.set_ylim(25, None)
    ax.set_xlim(3, 25.)

fig.supylabel('$y_{off}$ (mm)')
fig.supxlabel('$z_{off}$ (mm)')









plt.figure(figsize=(5, 4), constrained_layout=True)
plt.scatter((1 / df_to_plot[key_I_pe]), ((1 / df_to_plot[key_I_pc])), c='k')
# for i,pt in enumerate(selected_pts):
#     queried =query(*selected_pts[i,:]).iloc[0]
#     plt.scatter((1/queried[key_I_pe]),((1/queried[key_I_pc])),label = 'A%d'%(i+1),marker='*',s = 200,c =default_colors[i])
plt.xlabel("$1/I_{pe}$")
plt.ylabel("$1/I_{pc}$")
# plt.xscale('log')
# plt.yscale('log')
func_lg_1_Ipc = lambda lg_1_Ipe, k, b: -k * lg_1_Ipe + b
func_I_pc = lambda I_pe, a, b, c: a * I_pe ** (b) + c
func_1_Ipc = lambda _1_Ipe, A, B, C, D: A * (_1_Ipe + B) ** -C + D
from scipy.optimize import curve_fit

mask = (1 / df_to_plot[key_I_pe]) < 0.2  # ~pandas.isna(df_to_plot[key_runid])
# mask = ~pandas.isna(df_to_plot[key_runid])
(k, b), cov = curve_fit(func_lg_1_Ipc, numpy.log10(1 / df_to_plot[mask][key_I_pe]),
                        numpy.log10(1 / (df_to_plot[mask][key_I_pc])))
b, k = -2.8, 0.65
(A, B, C, D), cov = curve_fit(func_1_Ipc, (1 / df_to_plot[mask][key_I_pe]), (1 / (df_to_plot[mask][key_I_pc])),
                              p0=(0.055449999028688815, -0.001480053817205832, 0.547, -0.058716644183192924))
# a,b,c = [4.29,-0.547,0]
# (a,b,c),cov  = curve_fit(func_I_pc,  df_to_plot[mask][key_I_pe],df_to_plot[mask][key_I_pc],p0 = (a,b,c))
lg_1_Ipe = numpy.linspace(min(numpy.log10(1 / df_to_plot[key_I_pe])), max(numpy.log10(1 / df_to_plot[key_I_pe])), 100)
lg_1_Ipc = (func_lg_1_Ipc(lg_1_Ipe, k, b))
Ipe_linspace = numpy.linspace(min(df_to_plot[key_I_pe]), max(df_to_plot[key_I_pe]), 100)
_1_Ipe_linspace = numpy.linspace(min(1 / df_to_plot[key_I_pe]), max(1 / df_to_plot[key_I_pe]), 100)
plt.plot(10 ** lg_1_Ipe, 10 ** lg_1_Ipc, '--', label="$I_{pc} = %.2f I_{pe}^{%.2f}$" % (numpy.exp(-b), -k))
# plt.plot(10** lg_1_Ipe,func_1_Ipc(10** lg_1_Ipe,A,B,C,D),'--',label = "func_1_Ipc")
# plt.plot( 1/Ipe_linspace, 1/func_I_pc(Ipe_linspace, a,b,c),'--',label = "$I_{pc} = %.2f I_{pe}^{%.2f}$"%(numpy.exp(-b),-k))
plt.legend()
plt.xlim(0.0, 0.0016)
plt.ylim(-0.05, 1)
plt.gca().ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')

plt.scatter(*(1 / querieds[[key_I_pe, key_I_pc]]).values.T, marker='*', s=200,
            c='r')
for i in querieds.index:
    plt.annotate('A%d' % (i + 1), 1 / querieds.loc[i, [key_I_pe, key_I_pc]], (0.8, 0.5),
                 textcoords='offset fontsize',
                 # arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2',),
                 c='r')
plt.savefig("pareto_front.png", dpi=200)

key_Db = "Delta b"
df_to_plot[key_Db] = numpy.nan
mask = ~pandas.isna(df_to_plot[key_runid])
df_to_plot[key_Db][mask] = 1 / df_to_plot[key_I_pc] - func_1_Ipc((1 / df_to_plot[key_I_pe]), A, B, C,
                                                                 D)  # (numpy.log10(1 / df_to_plot[key_I_pc]) - func_lg_1_Ipc(numpy.log10(1 / df_to_plot[key_I_pe]), k, b))
# df_to_plot[key_Db][mask] =(numpy.log10(1 / df_to_plot[key_I_pc]) - func_lg_1_Ipc(numpy.log10(1 / df_to_plot[key_I_pe]), k, b))


# mask = (numpy.log10( 1/df_to_plot[key_I_pc]) - func_lg_1_Ipc(numpy.log10(1/df_to_plot[key_I_pe]), k, b)) > (0.0 * numpy.abs(b))
# indexes_worse  = df_to_plot.index[mask]
# plt.scatter(1/df_to_plot[mask][key_I_pe],1/df_to_plot[mask][key_I_pc])

# mask = (numpy.log10( 1/df_to_plot[key_I_pc]) - func_lg_1_Ipc(numpy.log10(1/df_to_plot[key_I_pe]), k, b)) < -(0.0 * numpy.abs(b))
# indexes_better  = df_to_plot.index[mask]
# plt.scatter(1/df_to_plot[mask][key_I_pe],1/df_to_plot[mask][key_I_pc])


plt.figure(figsize=(4, 4.2))
cf = plot_contour(df_to_plot, lambda df: df[key_Db], plt.gca(),
                  levels=numpy.linspace(-max(numpy.abs(df_to_plot[key_Db])), max(numpy.abs(df_to_plot[key_Db])), 31),
                  cmap='jet')
plt.gca().set_aspect("equal")
plt.gcf().colorbar(cf)
plt.xlabel('$z_{off}$ / mm')
plt.ylabel('$y_{off}$ / mm')
plt.title(r"$\Delta I_{pc}^{-1} = I_{pc}^{-1}-\hat I_{pc}^{-1}$")

cf = plt.contourf(Z, Y, griddata(df_Eabs[["z [mm]", "y [mm]", ]].values, I_pc, (Z, Y)),
                  locator=matplotlib.ticker.LogLocator())

cf = plt.contourf(Z, Y, griddata(df_Eabs[["z [mm]", "y [mm]", ]].values, I_pc, (Z, Y)),
                  locator=matplotlib.ticker.LogLocator())
# cf = plt.contourf(Z,Y, griddata(df_Eabs[["z [mm]","y [mm]",]].values, Eabs,(Z,Y)))
plt.scatter(*(df_to_plot[~pandas.isna(df_to_plot[key_runid])][[key_zoff, key_yoff]].values + [zoff0, yoff0]).T, c='k',
            alpha=1,
            s=2)
plt.xlabel('z / mm')
plt.ylabel('y / mm')
# plt.title("$I_{pc}$")
plt.title("$|E|_{max}$")
plt.gcf().colorbar(cf)
plt.savefig("E_abs.png", dpi=200)

import stl.mesh

mesh = stl.mesh.Mesh.from_file(r"E:\CSTprojects\rfCompressor\mesh.stl")

from mpl_toolkits import mplot3d

fig3d: plt.Figure = plt.figure()
ax3d = fig3d.add_axes(mplot3d.Axes3D(fig3d))
ax3d.add_collection3d(mplot3d.axes3d.art3d.Poly3DCollection(mesh.vectors))
scale = mesh.points.flatten()
ax3d.auto_scale_xyz(scale, scale, scale)
# plt.show()
