# -*- coding: utf-8 -*-
# @Time    : 2024/7/8 21:44
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : export_results.py
# @Software: PyCharm
import json
import os
import typing

import shapely.geometry

import cst.results
import matplotlib
import numpy
import pandas
import scipy
from scipy.interpolate import griddata

from _logging import logger

matplotlib.use('tkagg')
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt

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



f_target = 9.3e9


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
                                 key_parameter_combination: parameter_combination,key_result_dir:dir }
        logger.info(dir)
    return data


data = load_data(r"F:\changeworld\HPMCalc\theory\rfCompressor\test_SES_switch01.1.cst.results")
aaaa
cst_proj_path = r"E:\CSTprojects\rfCompressor\test_SES_switch01.1.paramsweep.cst"
# cst_proj_path = r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch01.1.cst"
proj: cst.results.ProjectFile = cst.results.ProjectFile(cst_proj_path,
                                                        allow_interactive=True)

# proj :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch02.cst",
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


for run_id in run_ids:
    S33_3D = numpy.array(proj.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3', run_id=run_id, ).get_data())
    S23_3D = numpy.array(proj.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3', run_id=run_id, ).get_data())
    S22_3D = numpy.array(proj.get_3d().get_result_item('1D Results\\S-Parameters\\S2,2', run_id=run_id, ).get_data())
    S32_3D = numpy.array(proj.get_3d().get_result_item('1D Results\\S-Parameters\\S3,2', run_id=run_id, ).get_data())
    # power_balance =(numpy.abs(S23_3D[:,1])**2+numpy.abs(S33_3D[:, 1])**2)
    sig_i3 = numpy.array(
        proj.get_schematic().get_result_item('Tasks\\Tran1\\TD Signals\\I3', run_id=run_id, ).get_data())
    sig_o23 = numpy.array(
        proj.get_schematic().get_result_item('Tasks\\Tran1\\TD Signals\\O2,3', run_id=run_id, ).get_data())
    sig_o33 = numpy.array(
        proj.get_schematic().get_result_item('Tasks\\Tran1\\TD Signals\\O3,3', run_id=run_id, ).get_data())

    df_S = pandas.DataFrame(data={
        key_freq: numpy.real(S22_3D[:, 0]),
        'S2,2': (S22_3D[:, 1]),
        key_S23: (S23_3D[:, 1]),
        'S3,2': (S32_3D[:, 1]),
        key_S33: (S33_3D[:, 1]),
        'Zref2/Ohm': S22_3D[:, 2],
        'Zref3/Ohm': S33_3D[:, 2],
        # 'Balance [3]': power_balance
    })
    signals = {'i3': signal_to_df(sig_i3),
               'o23': signal_to_df(sig_o23),
               'o33': signal_to_df(sig_o33),
               }
    result_dir = '%s.results/%04d' % (os.path.split(cst_proj_path)[1], run_id)
    os.makedirs(result_dir, exist_ok=True)
    signals_dir = '%s/signals' % result_dir
    os.makedirs(signals_dir, exist_ok=True)

    df_S.to_csv(r'%s/S-parameters.csv' % result_dir, index=False)
    for signal_name in signals:
        signals[signal_name].to_csv(r'%s/%s.csv' % (signals_dir, signal_name), index=False)
    parameter_combination = proj.get_schematic().get_parameter_combination(run_id)
    with open(r'%s/parameter_combination.json' % result_dir, 'w') as f:
        json.dump(parameter_combination, f)
    data[run_id] = {key_Sparameters: df_S, key_signals: signals, key_parameter_combination: parameter_combination,  key_result_dir: (result_dir)}
    # logger.info(r'"%s" done' % (os.path.abspath('%s/parameter_combination.json' % result_dir)))


def func_o(t, omega, Q, Delta_t, o_inf):
    return numpy.piecewise(t, [t > Delta_t, ], [lambda t: o_inf * (1 - numpy.exp(-(t - Delta_t) / (2 * Q / omega))), 0])


omega0 = 2 * numpy.pi * f_target
__func_o = lambda t, Q, Delta_t, o_inf: func_o(t, omega0, Q, Delta_t, o_inf)

key_zoff = 'SwitchZoffset'
key_yoff = 'SwitchRoffset'
key_normalized_Pout_inf = 'normalized_Pout_inf'
key_runid = 'run_id'
key_signal_fft_at_f_target = "signal_fft_at_f_target"
key_I_pe = 'I_pe'
key_I_pc = 'I_pc'


sampled_time = data[list(data.keys())[0]][key_signals]['o23'][key_time]

resampled_time = numpy.linspace(sampled_time[0], sampled_time.values[-1], len(sampled_time))
resampled_dt = numpy.diff(resampled_time)[0]  # ns
freqs = scipy.fft.fftfreq(len(sampled_time), resampled_dt * 1e-9)  # unit in Hz

df_to_plot = pandas.DataFrame(
    columns=[key_runid, key_zoff, key_yoff, 'Q', key_normalized_Pout_inf, key_signal_fft_at_f_target,
             key_power_balance,key_S23,key_result_dir,key_I_pe,key_I_pc])
for run_id in data:
    df_o23 = data[run_id][key_signals]['o23']
    trusted_index = (df_o23[key_time] > 1)
    o23_inf = (df_o23[trusted_index][key_interpolated_periodic_avg_square].values[-2] * 2) ** 0.5

    # (Q, Delta_t, o23_inf), cov = curve_fit(__func_o, df_o23[trusted_index][key_time].values * 1e-9,
    #                                        (df_o23[trusted_index][
    #                                             key_interpolated_periodic_avg_square].values * 2) ** 0.5,
    #                                        p0=[100, 1e-9, o23_inf], bounds=numpy.array([[1, 2e4], [0, 2e-9], [0, 1]]).T)
    Q = 1.0
    resampled_signal = numpy.interp(resampled_time, df_o23[key_time], df_o23[key_signals])
    signal_fft = numpy.abs(scipy.fft.fft(resampled_signal))
    # plt.plot(freqs,signal_fft,label = '%04d'%run_id)
    signal_fft_at_f_target = numpy.interp(f_target, freqs, signal_fft)
    # plt.figure()
    # plt.plot(df_o23[key_time].values * 1e-9, df_o23[key_signals].values)
    # plt.plot(df_o23[key_time].values * 1e-9, __func_o(df_o23[key_time].values * 1e-9,Q, Delta_t, o23_inf))

    # S23_,S33_ = numpy.interp( f_target / 1e9, data[run_id][key_Sparameters][key_freq].values, numpy.vstack( [ data[run_id][key_Sparameters][key_S23],data[run_id][key_Sparameters][key_S33]])  )
    from scipy.interpolate import interp1d

    S23_, S33_ = interp1d(data[run_id][key_Sparameters][key_freq].values, numpy.vstack(
        [data[run_id][key_Sparameters][key_S23], data[run_id][key_Sparameters][key_S33]]))(f_target / 1e9)

    df_to_plot.loc[len(df_to_plot)] = {
        key_runid: run_id,
        key_zoff: data[run_id][key_parameter_combination][key_zoff],
        key_yoff: data[run_id][key_parameter_combination][key_yoff],
        'Q': Q,
        key_normalized_Pout_inf: o23_inf ** 2,
        key_signal_fft_at_f_target: signal_fft_at_f_target,
        key_power_balance: (numpy.abs(S23_) ** 2 + numpy.abs(S33_) ** 2) ** 0.5,
        key_S23: numpy.abs(S23_),
        key_result_dir : data[run_id][key_result_dir]
    }
# fake_data = pandas.DataFrame(columns=df_to_plot.columns)
# fake_data[[key_zoff,key_yoff]] =numpy.array(numpy.meshgrid(numpy.linspace(0, 5.4, 5), numpy.linspace(23.7, 34.4, 5))).transpose((1, 2, 0)).reshape(
#         (-1, 2))
#
# fake_data = pandas

# df_to_plot_  = df_to_plot.copy()


df_to_plot = df_to_plot[~pandas.isna(df_to_plot[key_runid])]

df_to_plot = pandas.concat([df_to_plot,
                            pandas.DataFrame(numpy.vstack([
                                numpy.array(
                                    numpy.meshgrid(numpy.linspace(0, 5.4, 5), numpy.linspace(24, 33, 5))).transpose(
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

def query(zoff, yoff, EPS=0.5):
    # EPS=0.5
    return df_to_plot[(numpy.abs(df_to_plot[key_zoff] - zoff) < EPS) & (numpy.abs(df_to_plot[key_yoff] - yoff) < EPS)]


# plt.figure()
# plt.scatter(*df_to_plot[[key_zoff, key_yoff]].values.T)
df_seq1 = df_to_plot[df_to_plot[key_yoff] > 34]
df_seq2 = df_to_plot[(df_to_plot[key_yoff] > 22) & (df_to_plot[key_yoff] < 36) & (df_to_plot[key_zoff] >= 6) & (
        df_to_plot[key_zoff] <= 23)]
df_seq3 = df_to_plot[df_to_plot[key_yoff] < 24]

# df_seqs = [df_seq1, df_seq2, df_seq3]
df_seqs = [df_to_plot]



def plot_contour(df_seq: pandas.DataFrame, how_to_process_df_seq, ax: plt.Axes, **kwargs):
    Z, Y = numpy.meshgrid(numpy.linspace(min(df_seq[key_zoff]), max(df_seq[key_zoff]), 100),
                          numpy.linspace(min(df_seq[key_yoff]), max(df_seq[key_yoff]), 100))
    cf = ax.contourf(Z, Y, griddata(df_seq[[key_zoff, key_yoff]].values, how_to_process_df_seq(df_seq), (Z, Y),
                                    # method='cubic'
                                    ), **kwargs)
    return cf

def queried_to_table_in_paper (df_to_plot:pandas.DataFrame):
    new_df=pandas.DataFrame()
    new_df["case"] =["A%d"%(i+1) for i in  df_to_plot.index]
    new_df["zoff"] = df_to_plot[ key_zoff]
    new_df["yoff"] = df_to_plot[ key_yoff]
    new_df["S21"] = df_to_plot[ key_S23]
    new_df["1/(1-sum)"] = 1 / (1 - df_to_plot[ key_power_balance] ** 2)
    new_df["Ipe"] = df_to_plot[key_I_pe]
    new_df["Ipc"] = df_to_plot[key_I_pc]
    return new_df






selected_pts = numpy.array([
    [2.997, 15.75],
    [10.8,22.8],
    [15.9,26.4],
    # [17.97,28.92],
    [18.8, 30.0],
    # [19.8,32.2],
    # [19.3,35.7],
    # [10.5, 38.8],
    # [20.4, 27.6],
    # [21.0, 41.8],
])
querieds = pandas.DataFrame(columns=df_to_plot.columns)
queried_to_table_in_paper(querieds).to_csv("temp.csv")




fig: plt.Figure = plt.figure( constrained_layout=True,figsize=(5,5))
gs = plt.GridSpec(2,2,figure = fig, )
axs = [fig.add_subplot(gs_,) for gs_ in  (gs[0,0],gs[0,1],gs[1,0],gs[1,1])]
# df_to_plot[key_I_pe] =(lambda  df_seq:df_seq[key_S23] ** 2* df_seq[key_power_balance] ** 2/ (1. - df_seq[key_power_balance] ** 2))(df_to_plot)
df_to_plot[key_I_pe] = df_to_plot[key_S23]**2*df_to_plot[key_power_balance]**4/(1-df_to_plot[key_power_balance]**2)**2
for ax in axs[1:]:
    ax.sharex(axs[0])
    ax.sharey(axs[0])

for df_seq in df_seqs:
    f1 = lambda df_seq:  df_seq[key_I_pe]
    cf1 = plot_contour(df_seq, f1, axs[0], levels=20)
    f1 = lambda df_seq:  df_seq[key_I_pe]
    cf2 = plot_contour(df_seq, f1, axs[1], levels=20)
    f1 = lambda df_seq: df_seq[key_S23]
    cf3 = plot_contour(df_seq, f1, axs[2], levels=numpy.linspace(0, 1, 20)
                       # numpy.linspace(min(f1(df_to_plot)), max(f1(df_to_plot)), 100)
                       )
    f1 = lambda df_seq: 1. / (1. - df_seq[key_power_balance] ** 2)  # df_seq[key_signal_fft_at_f_target]
    cf4 = plot_contour(df_seq, f1, axs[3], levels=20)
# titles = ['$P_{out}(\infty) / Q$', '$P_{out}(\infty) $', '$1 / Q$', '$F(o_{23})|_{%.1fGHz} P_{out}(\infty)$'%(f_target/1e9)]
titles = ['$I_{pe}$','$I_{pe}$' , '$|S_{2,1}|$',
          '$1 / (1-\sum_{i}|S_{i,1}|^2)$' ]

circ = shapely.geometry.Point(13.74, 33.13, ).buffer(5.9)
_scater_size = [0,0.01,0.01,0.01]
for i, ax in enumerate(axs):
    ax: plt.Axes
    ax.scatter(*df_to_plot[~pandas.isna(df_to_plot[key_runid])][[key_zoff, key_yoff]].values.T, c='k', alpha=1,
               s=_scater_size[i])
    # ax.scatter(*df_to_plot[pandas.isna(df_to_plot[key_runid])][[key_zoff, key_yoff]].values.T, alpha=1, s=1)
    # ax.scatter(*numpy.array(circ.exterior.xy)[:,::4], alpha=1,s = _scater_size)
    # ax.scatter(*circ.centroid.xy, alpha=1,s = _scater_size)
    ax.set_aspect('equal')
    ax.set_title(titles[i])
    ax.set_xlabel('$z_{off}$ / mm')
    ax.set_ylabel('$y_{off}$ / mm')
for cf in [cf1, cf2,cf3, cf4]:
    fig.colorbar(cf)
default_colors = list(mcolors.TABLEAU_COLORS)


for i,pt in enumerate(selected_pts):
    querieds.loc[len(querieds)] = query(*selected_pts[i,:]).iloc[0]
axs[1].scatter(*querieds[[key_zoff,key_yoff]].values.T, marker='*', s=5,c = 'r' )
for i in querieds.index:
    axs:typing.List[plt.Axes]
    axs[1].annotate('A%d' % (i + 1),querieds.loc[i,[key_zoff,key_yoff]],(-1.5,0.02),
                 textcoords='offset fontsize',
                    c = 'r')
# for i, ax in enumerate(axs):
#     ax: plt.Axes
#     for selected_pt in selected_pts:
#         queried = query(*selected_pt).iloc[0]
#         ax.scatter(*queried[[key_zoff,key_yoff]].values,s = 2)
# df_to_plot.to_csv('df_to_plot.csv',index=False)
# plt.savefig('contour_Pinf.pdf')
plt.savefig('I_pe.eps', dpi=200)








figsize  = (3,7)
fig, axs = plt.subplots(len(selected_pts),1, sharex = True, sharey = True, constrained_layout = True,figsize = figsize)
fig2, axs2 = plt.subplots(len(selected_pts),1, sharex = True, sharey = True, constrained_layout = True,figsize = figsize)
fig2:plt.Figure
s_signals ="/signals"
for i,selected_pt in enumerate(selected_pts):
    queried = query(*selected_pt).iloc[0]
    result_dir = queried[key_result_dir]
    logger.info(result_dir)
    df_signals = {'i3':pandas.read_csv(result_dir +s_signals+ '/i3.csv'),
           'o23': pandas.read_csv(result_dir+s_signals + '/o23.csv'),
           }
    df_o23 = df_signals['o23']
    df_Sparams = pandas.read_csv(result_dir+'/S-parameters.csv'                                 )
    axs[i].plot(  df_Sparams[key_freq],numpy.abs( df_Sparams[key_S33].astype(complex) ),label = "S11")
    axs[i].plot(  df_Sparams[key_freq], numpy.abs(df_Sparams[key_S23].astype(complex)) ,label = "S21")
    axs[i].legend()
    axs2[i].plot( df_o23[key_time],  df_o23['signal'] ,label = "A%d (%.2f, %.2f)"%(i+1, queried[key_zoff], queried[key_yoff])  ,lw= 0.2                )
    # axs2[i].legend()
    # axs2[i].grid()
axs[-1].set_xlabel ("frequency / GHz")
axs[-1].set_xlim (9, 9.6)
axs2[-1].set_xlim(0,20)
axs2[-1].set_xlabel ("time / ns")
fig2.supylabel(r"normalized output signal / $\sqrt{W}$")
# axs2[2].set_ylabel (r"normalized output signal / $\sqrt{W}$")
fig.savefig('S-params.png',dpi = 200)
fig2.savefig('signal.png',dpi = 200)



Z,Y =numpy.meshgrid( numpy.linspace(244,305,100),numpy.linspace(-17,37,50))
df_ptlist = pandas.DataFrame({'x':numpy.zeros(Z.ravel().shape)+1 ,'y':Y.ravel(), 'z':Z.ravel(),})
df_ptlist.to_csv(r"E:\CSTprojects\rfCompressor\ptlist.txt",index= False,sep = ' ',header=None
                 )

df_Eabs = pandas.read_csv(r"E:\CSTprojects\rfCompressor\test_SES_switch01.1-energy_storage\Export\3d\e-field (f=f0) [3].txt",sep=r'\s\s+',skiprows=lambda idx:idx==1)# pandas.read_csv(r"E:\CSTprojects\rfCompressor\Eabs.txt",sep=r'\s\s+',skiprows=lambda idx:idx==1)
# Z,Y = numpy.meshgrid(numpy.linspace(min(df_Eabs["z [mm]"]),max(df_Eabs["z [mm]"]),1000),numpy.linspace(min(df_Eabs["y [mm]"]),max(df_Eabs["y [mm]"]),500))


# plt.scatter(*df_Eabs[["z [mm]","y [mm]",]].values.T)



colnames_E = []
for col in df_Eabs.columns:
    if col.startswith("E"):
        colnames_E.append(col)
Eabs = (df_Eabs[colnames_E]**2).sum(axis = 1)**0.5
Eabs_max =Eabs.max()
df_to_plot[key_I_pc] = numpy.nan
zoff0, yoff0 = 266.71,-13.4

r_tube = 3
pts_to_calc_tube_max = numpy.array((numpy.meshgrid(numpy.linspace(-r_tube, r_tube,50),numpy.linspace(-r_tube, r_tube,50)))).transpose((1,2,0)).reshape((-1, 2))
pts_to_calc_tube_max = pts_to_calc_tube_max[pts_to_calc_tube_max[:,0]**2+pts_to_calc_tube_max[:,1]**2<=r_tube**2]
# plt.figure()
# plt.scatter( *pts_to_calc_tube_max.T)

from scipy.interpolate import LinearNDInterpolator
Eabs_interpolator = LinearNDInterpolator( df_Eabs[["z [mm]","y [mm]",]].values, Eabs)
for i in df_to_plot[~pandas.isna(df_to_plot[key_runid])].index:
    zoff,yoff = df_to_plot[[key_zoff,key_yoff]].loc[i]
    Eabs_max_tube  =( Eabs_interpolator(numpy.array([zoff0+zoff,yoff0+yoff])+pts_to_calc_tube_max)).max()
    df_to_plot.loc[i,key_I_pc] = (Eabs_max / Eabs_max_tube)**2

# from matplotlib import colors
# norm = colors.LogNorm(vmin=Z.min(), vmax=Z.max())
plt.figure(constrained_layout = True,figsize=(3,3.5))
plt.gca().set_aspect('equal')
cf = plot_contour(df_to_plot,lambda df: numpy.log10(df[key_I_pc]) ,plt.gca(),#locator=matplotlib.ticker.LogLocator(),
                  levels = 21,zorder = -1)
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

plt.scatter(*querieds[[key_zoff,key_yoff]].values.T, marker='*', s=10,c = 'r' )
for i in querieds.index:
    plt.annotate('A%d' % (i + 1),querieds.loc[i,[key_zoff,key_yoff]],(0.3,0.0),
                 textcoords='offset fontsize',
                    c = 'r')
plt.savefig("I_pc.eps",dpi = 200)


plt.figure(figsize=(5,4),constrained_layout = True)
plt.scatter((1/df_to_plot[key_I_pe]),((1/df_to_plot[key_I_pc])),c = 'k')
# for i,pt in enumerate(selected_pts):
#     queried =query(*selected_pts[i,:]).iloc[0]
#     plt.scatter((1/queried[key_I_pe]),((1/queried[key_I_pc])),label = 'A%d'%(i+1),marker='*',s = 200,c =default_colors[i])
plt.xlabel("$1/I_{pe}$")
plt.ylabel("$1/I_{pc}$")
# plt.xscale('log')
# plt.yscale('log')
func_lg_1_Ipc = lambda lg_1_Ipe, k, b : -k * lg_1_Ipe + b
func_I_pc = lambda I_pe ,a,b,c:a*I_pe**(b) +c
func_1_Ipc = lambda _1_Ipe ,A,B,C,D:A*(_1_Ipe+B)**-C+D
from scipy.optimize import curve_fit
mask =(1/df_to_plot[key_I_pe])<0.2 #~pandas.isna(df_to_plot[key_runid])
# mask = ~pandas.isna(df_to_plot[key_runid])
(k, b),cov  = curve_fit(func_lg_1_Ipc, numpy.log10(1 / df_to_plot[mask][key_I_pe]), numpy.log10(1 / (df_to_plot[mask][key_I_pc])))
b,k = -2.8,0.65
(A,B,C,D),cov  = curve_fit(func_1_Ipc,(1 / df_to_plot[mask][key_I_pe]),(1 / (df_to_plot[mask][key_I_pc])),p0=  (0.055449999028688815, -0.001480053817205832,0.547, -0.058716644183192924))
# a,b,c = [4.29,-0.547,0]
# (a,b,c),cov  = curve_fit(func_I_pc,  df_to_plot[mask][key_I_pe],df_to_plot[mask][key_I_pc],p0 = (a,b,c))
lg_1_Ipe = numpy.linspace(min(numpy.log10(1/df_to_plot[key_I_pe])),max(numpy.log10(1/df_to_plot[key_I_pe])),100)
lg_1_Ipc = (func_lg_1_Ipc(lg_1_Ipe, k, b))
Ipe_linspace=numpy.linspace(min(df_to_plot[key_I_pe]),max(df_to_plot[key_I_pe]),100)
_1_Ipe_linspace=numpy.linspace(min(1/df_to_plot[key_I_pe]),max(1/df_to_plot[key_I_pe]),100)
plt.plot( 10** lg_1_Ipe, 10**lg_1_Ipc,'--',label = "$I_{pc} = %.2f I_{pe}^{%.2f}$"%(numpy.exp(-b),-k))
# plt.plot(10** lg_1_Ipe,func_1_Ipc(10** lg_1_Ipe,A,B,C,D),'--',label = "func_1_Ipc")
# plt.plot( 1/Ipe_linspace, 1/func_I_pc(Ipe_linspace, a,b,c),'--',label = "$I_{pc} = %.2f I_{pe}^{%.2f}$"%(numpy.exp(-b),-k))
plt.legend()
plt.xlim(0.0,0.0016)
plt.ylim(-0.05,1)
plt.gca().ticklabel_format(style='sci', scilimits=(-1,2), axis='x')

plt.scatter(*(1/querieds[[key_I_pe,key_I_pc]]).values.T, marker='*', s=200,
            c = 'r' )
for i in querieds.index:
    plt.annotate('A%d' % (i + 1),1/querieds.loc[i,[key_I_pe,key_I_pc]],(0.8,0.5),
                 textcoords='offset fontsize',
                    # arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2',),
                    c = 'r')
plt.savefig("pareto_front.png",dpi = 200)






key_Db = "Delta b"
df_to_plot[key_Db] = numpy.nan
mask = ~pandas.isna(df_to_plot[key_runid])
df_to_plot[key_Db][mask] =1/df_to_plot[key_I_pc] - func_1_Ipc( (1 / df_to_plot[key_I_pe]), A,B,C,D)  #(numpy.log10(1 / df_to_plot[key_I_pc]) - func_lg_1_Ipc(numpy.log10(1 / df_to_plot[key_I_pe]), k, b))
# df_to_plot[key_Db][mask] =(numpy.log10(1 / df_to_plot[key_I_pc]) - func_lg_1_Ipc(numpy.log10(1 / df_to_plot[key_I_pe]), k, b))


# mask = (numpy.log10( 1/df_to_plot[key_I_pc]) - func_lg_1_Ipc(numpy.log10(1/df_to_plot[key_I_pe]), k, b)) > (0.0 * numpy.abs(b))
# indexes_worse  = df_to_plot.index[mask]
# plt.scatter(1/df_to_plot[mask][key_I_pe],1/df_to_plot[mask][key_I_pc])

# mask = (numpy.log10( 1/df_to_plot[key_I_pc]) - func_lg_1_Ipc(numpy.log10(1/df_to_plot[key_I_pe]), k, b)) < -(0.0 * numpy.abs(b))
# indexes_better  = df_to_plot.index[mask]
# plt.scatter(1/df_to_plot[mask][key_I_pe],1/df_to_plot[mask][key_I_pc])





plt.figure(figsize=(4,4.2))
cf = plot_contour(df_to_plot, lambda df:df[key_Db], plt.gca(), levels = numpy.linspace(-max(numpy.abs(df_to_plot[key_Db])),max(numpy.abs(df_to_plot[key_Db])),31),cmap = 'jet')
plt.gca().set_aspect("equal")
plt.gcf().colorbar(cf)
plt.xlabel('$z_{off}$ / mm')
plt.ylabel('$y_{off}$ / mm')
plt.title(r"$\Delta I_{pc}^{-1} = I_{pc}^{-1}-\hat I_{pc}^{-1}$")



cf = plt.contourf(Z,Y, griddata(df_Eabs[["z [mm]","y [mm]",]].values, I_pc,(Z,Y)),locator=matplotlib.ticker.LogLocator())

cf = plt.contourf(Z,Y, griddata(df_Eabs[["z [mm]","y [mm]",]].values, I_pc,(Z,Y)),locator=matplotlib.ticker.LogLocator())
# cf = plt.contourf(Z,Y, griddata(df_Eabs[["z [mm]","y [mm]",]].values, Eabs,(Z,Y)))
plt.scatter(*(df_to_plot[~pandas.isna(df_to_plot[key_runid])][[key_zoff, key_yoff]].values+[zoff0, yoff0]).T, c='k', alpha=1,
           s=2)
plt.xlabel('z / mm')
plt.ylabel('y / mm')
# plt.title("$I_{pc}$")
plt.title("$|E|_{max}$")
plt.gcf().colorbar(cf)
plt.savefig("E_abs.png",dpi = 200)

import stl.mesh
mesh = stl.mesh.Mesh.from_file(r"E:\CSTprojects\rfCompressor\mesh.stl")

from mpl_toolkits import mplot3d
fig3d:plt.Figure = plt.figure()
ax3d =fig3d.add_axes(mplot3d.Axes3D(fig3d))
ax3d.add_collection3d(mplot3d.axes3d.art3d.Poly3DCollection(mesh.vectors))
scale = mesh.points.flatten()
ax3d.auto_scale_xyz(scale,scale,scale)
# plt.show()


