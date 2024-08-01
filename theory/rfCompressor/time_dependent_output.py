# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 18:56
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : time_dependent_output.py
# @Software: PyCharm
import cst.results
import cst.results
import matplotlib
import numpy
import pandas
import scipy
import shapely.lib
from scipy.fft import fft, fftfreq, ifft
from _logging import logger
from scipy.optimize import curve_fit
from shapely.geometry import LineString

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_target = 9.3e9


def to_df(o23):
    df_o23 = pandas.DataFrame(o23)
    df_o23[key_period_for_calculate_avg] = df_o23[0] * 1e-9 // (2 / f_target)
    df_o23[key_square] = df_o23[1] ** 2
    df_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
    df_o23[key_interpolated_periodic_avg_square] = numpy.interp(df_o23[0], df_time_avg[0], df_time_avg['square'], )
    df_o23[key_complex] = scipy.signal.hilbert(df_o23[1])
    return df_o23


# nw = skrf.network.Network(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch01.1.s2p")
# vf = skrf.VectorFitting(nw)
# vf.auto_fit()
# plt.figure()
# ax =vf.plot_s_mag()

# 需包含放能阶段的S参数
proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
    r"E:\CSTprojects\rfCompressor\test_SES_switch01.1 - copy.cst",
    allow_interactive=True)
proj_3D_charging: cst.results.ProjectFile = cst.results.ProjectFile(
    r"E:\CSTprojects\rfCompressor\test_SES_switch01.1-energy_storage.cst",
    allow_interactive=True)

# proj :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch02.cst",
#                                                         allow_interactive=True)
# proj_3D_charging :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch02.1.cst",
#                                                         allow_interactive=True)
S33_3D = numpy.array(proj_3D.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3', ).get_data())
S23_3D = numpy.array(proj_3D.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3', ).get_data())
Zref_2 = numpy.array(proj_3D.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 2(1)').get_data())
Zref_3 = numpy.array(proj_3D.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 3(1)').get_data())
Zref_3_interpolator = scipy.interpolate.interp1d(numpy.real(Zref_3[:, 0]), numpy.real(Zref_3[:, 1]),
                                                 fill_value="extrapolate")
Zref_2_interpolator = scipy.interpolate.interp1d(numpy.real(Zref_2[:, 0]), numpy.real(Zref_2[:, 1]),
                                                 fill_value="extrapolate")

S33_3D_interpolator = lambda f: numpy.piecewise(f, [numpy.real(f) >= 0, ],
                                                [lambda f: numpy.interp(numpy.real(f), numpy.real(S33_3D[:, 0]),
                                                                        S33_3D[:, 1], left=complex(0),
                                                                        right=complex(0)),
                                                 lambda f: numpy.interp(-numpy.real(f), numpy.real(S33_3D[:, 0]),
                                                                        numpy.conj(S33_3D[:, 1]), left=complex(0),
                                                                        right=complex(0))
                                                 ], )
S23_3D_interpolator = lambda f: numpy.piecewise(f, [numpy.real(f) >= 0, ],
                                                [lambda f: numpy.interp(numpy.real(f), numpy.real(S23_3D[:, 0]),
                                                                        S23_3D[:, 1], left=0, right=0),
                                                 lambda f: numpy.interp(-numpy.real(f), numpy.real(S23_3D[:, 0]),
                                                                        numpy.conj(S23_3D[:, 1]), left=0, right=0)
                                                 ])

S33_3D_charging = numpy.array(proj_3D_charging.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3', ).get_data())
S23_3D_charging = numpy.array(proj_3D_charging.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3', ).get_data())
Zref_2_charging = numpy.array(
    proj_3D_charging.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 2(1)').get_data())
Zref_3_charging = numpy.array(
    proj_3D_charging.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 3(1)').get_data())
Zref_3_interpolator_charging = scipy.interpolate.interp1d(numpy.real(Zref_3_charging[:, 0]),
                                                          numpy.real(Zref_3_charging[:, 1]), fill_value="extrapolate")
Zref_2_interpolator_charging = scipy.interpolate.interp1d(numpy.real(Zref_2_charging[:, 0]),
                                                          numpy.real(Zref_2_charging[:, 1]), fill_value="extrapolate")

S33_3D_interpolator_charging = lambda f: numpy.piecewise(f, [numpy.real(f) >= 0, ],
                                                         [lambda f: numpy.interp(numpy.real(f),
                                                                                 numpy.real(S33_3D_charging[:, 0]),
                                                                                 S33_3D_charging[:, 1], left=complex(0),
                                                                                 right=complex(0)),
                                                          lambda f: numpy.interp(-numpy.real(f),
                                                                                 numpy.real(S33_3D_charging[:, 0]),
                                                                                 numpy.conj(S33_3D_charging[:, 1]),
                                                                                 left=complex(0), right=complex(0))
                                                          ], )
S23_3D_interpolator_charging = lambda f: numpy.piecewise(f, [numpy.real(f) >= 0, ],
                                                         [lambda f: numpy.interp(numpy.real(f),
                                                                                 numpy.real(S23_3D_charging[:, 0]),
                                                                                 S23_3D_charging[:, 1], left=0,
                                                                                 right=0),
                                                          lambda f: numpy.interp(-numpy.real(f),
                                                                                 numpy.real(S23_3D_charging[:, 0]),
                                                                                 numpy.conj(S23_3D_charging[:, 1]),
                                                                                 left=0, right=0)
                                                          ])

# t = numpy.linspace(-10e-9, 40e-9, 100000)
dt = 5e-13#t[1] - t[0]

# Dt_left = 0.52e-9


# def get_signal(time:numpy.ndarray, signal:numpy.ndarray,Dt = 0):
#     """
#     :param signal:
#     :param Dt:
#     :return: signal(t+Dt)
#     """
#
#     new_signal = numpy.interp(time - Dt, time, signal )
#     return new_signal

# i3 = numpy.piecewise(t, [(t > 0) & (t < 10e-9), ], [lambda t: numpy.sin(2 * numpy.pi * f_target * t), 0])
# i3_fft = fft(i3)
Nfreqs = 100000
freqs = fftfreq(Nfreqs, dt)

impulse_response_23 = ifft(S23_3D_interpolator(freqs.astype(complex) / 1e9))[:Nfreqs // 2]
impulse_response_33 = ifft(S33_3D_interpolator(freqs.astype(complex) / 1e9))[:Nfreqs // 2]

impulse_response_23_charging = ifft(S23_3D_interpolator_charging(freqs.astype(complex) / 1e9))[:Nfreqs // 2]
impulse_response_33_charging = ifft(S33_3D_interpolator_charging(freqs.astype(complex) / 1e9))[:Nfreqs // 2]
angle_S33_charging =numpy.angle(S33_3D_interpolator_charging(complex(f_target) /1e9))#+0.7
# 修正相位：在一个RF周期内微调，使其满足S参数的约束，而几乎不影响幅值。
T_RF = (1/f_target)
Dt_lefts = numpy.array((*numpy.arange(0.2e-9, 1e-9, 0.2e-9),*numpy.arange(1e-9,10e-9,1e-9),20e-9))
Dt_lefts =( Dt_lefts//T_RF)*T_RF + angle_S33_charging/(2*numpy.pi) * T_RF

recorded_o23 = []
t0 = -max(Dt_lefts)*2

def get_ts(signal: numpy.ndarray, t_start=t0, dt=dt):
    N = len(signal)
    return numpy.linspace(t_start, t_start + N * dt, N)
for i, Dt_left in enumerate(Dt_lefts):

    t = numpy.arange(t0, 0 , dt)

    i3_charging = numpy.piecewise(t, [(t >t0# -10e-9
                                       ) & (t < 0), ], [lambda t: numpy.sin(2 * numpy.pi * f_target * t), 0])
    # i3_discharging =numpy.piecewise(t, [(t>0),], [lambda t:numpy.sin(2*numpy.pi*f_target*(t)) ,0])

    # plt.figure()
    o33_from_convolve_charging = scipy.signal.convolve(i3_charging, impulse_response_33_charging, )
    o33 = o33_from_convolve_charging
    while t[-1] < 100e-9:
        i3_discharging = numpy.piecewise(t, [t > 0, ], [lambda t: numpy.interp(t - Dt_left, get_ts(o33), o33), 0])
        o33_from_convolve_discharging = scipy.signal.convolve(i3_discharging, impulse_response_33, )
        if len(o33_from_convolve_discharging) - len(o33_from_convolve_charging) > 0:
            o33 = numpy.pad(o33_from_convolve_charging,
                            (0, len(o33_from_convolve_discharging) - len(o33_from_convolve_charging)), 'constant',
                            constant_values=(0, 0)) + o33_from_convolve_discharging
        else:
            o33 = o33_from_convolve_charging[:len(o33_from_convolve_discharging)] + o33_from_convolve_discharging
        # i3_charging = numpy.hstack([i3_charging,o33_from_convolve_charging[:len(t)][-int(Dt_left//dt):]])
        # i3_discharging = numpy.hstack([i3_discharging,o33[:len(t)][-int(Dt_left//dt):]])
        t = numpy.arange(t[0], t[-1] + Dt_left, dt)
        # o33 = scipy.signal.convolve(i3_discharging, impulse_response_33, )
    o23_charging = scipy.signal.convolve(i3_charging, impulse_response_23_charging, )
    o23_discharging =scipy.signal.convolve(i3_discharging, impulse_response_23, )
    o23 =  numpy.pad(o23_charging,
                            (0, len(o23_discharging) - len(o23_charging)), 'constant',
                            constant_values=(0, 0)) + o23_discharging
    df_i3_discharging = to_df(numpy.vstack((get_ts(i3_discharging) * 1e9, i3_discharging)).T.astype(float))
    df_o33 = to_df(numpy.vstack((get_ts(o33,)* 1e9, o33.real)).T.astype(float))
    df_o23 = to_df(numpy.vstack((get_ts(o23) * 1e9, o23.real)).T.astype(float))
    # plt.figure()
    # plt.plot(get_ts(i3_charging), i3_charging)
    # plt.plot(get_ts(o33_from_convolve_charging),o33_from_convolve_charging)
    # plt.plot(get_ts(o33_from_convolve_discharging),o33_from_convolve_discharging)
    # plt.plot(get_ts(o33_from_convolve_charging),o33)
    # plt.plot(get_ts(i3_discharging),i3_discharging)
    recorded_o23.append(df_o23)
    logger.info(i)

trusted_signal =lambda df:( df[0]>0)
eta_OM_max = []

for i, df_o23 in enumerate(recorded_o23):
    eta_OM_max.append(2 * df_o23[trusted_signal(df_o23)][key_interpolated_periodic_avg_square].max())
eta_OM_max = numpy.array(eta_OM_max)

plt.figure()
# plt.plot(t,i3_charging)
plt.plot(get_ts(i3_discharging), i3_discharging, label='i3')
plt.plot(get_ts(o33), o33, label='o33')
plt.plot(get_ts(o23), o23, label='o23')
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



import scipy.constants as C
gammadata = numpy.array(proj_3D_charging.get_3d().get_result_item('1D Results\\Port Information\\Gamma\\3(1)').get_data())
v_p = 2*numpy.pi *f_target / numpy.interp((f_target/1e9),gammadata[:,0].real,gammadata[:,1]).imag
v_g =  C.c**2 /v_p

plt.figure(figsize=(5,4),constrained_layout = True)
for i,df_o23_ in enumerate(recorded_o23):
    if i%2==0 or False :
        # plt.plot(df_o23_[0],(numpy.abs(df_o23_[key_complex])**2),label = '$\Delta t$ = %.1f ns ($L_1$ = %.2f m)'%(Dt_lefts[i]/1e-9, Dt_lefts[i]*v_g/2))
        plt.plot(df_o23_[0],df_o23_[key_interpolated_periodic_avg_square]*2,label = '$\Delta t$ = %.2f ns ($L_1$ = %.2f m)'%(Dt_lefts[i]/1e-9, Dt_lefts[i]*v_g/2))
        # plt.plot(df_o23_[0],(numpy.abs(df_o23_[key_complex])**2),label = '$L_1$ = %.2f m'%( Dt_lefts[i]*v_g/2))
plt.xlabel('time (ns)')
plt.ylabel(r'$\eta_{OM}(t)$')
# plt.ylabel(r'normalized amplitude / $\sqrt{W}$')
plt.legend()
plt.savefig("eta_OM_of_different_Dt.eps")


G_OM =1/( 1-(numpy.abs(S23_3D_interpolator_charging(complex(f_target/1e9)))**2+ numpy.abs(S33_3D_interpolator_charging(complex(f_target/1e9)))**2))
L_OM = (299.77-240)*1e-3

def func_G_cav_peak(Dt_lefts ):return  G_OM*2*L_OM/v_g/(Dt_lefts+2*L_OM/v_g)
G_cav_peak = func_G_cav_peak(Dt_lefts)
_func_eta_OM_peak= lambda Dt_left,eta_inf, tau, Dt_right:numpy.piecewise(
    Dt_left, [Dt_left +Dt_right>0],( lambda Dt_left :eta_inf*(1-numpy.exp(-(Dt_left+Dt_right)/tau))**2,0))
df = pandas.read_csv(r"F:\changeworld\HPMCalc\theory\rfCompressor\test_SES_switch01.1.cst.results\0218\signals\o23.csv")

t_ = numpy.arange(0, 40e-9, dt)
test_01 = scipy.signal.convolve(impulse_response_23,numpy.sin(2*numpy.pi *f_target *t_))
df_test_01 = to_df(numpy.array((1e9 * t_, test_01[:len(t_)])).T.real)
# df_test_01_interpolator = scipy.interpolate.interp1d
sig_o23 = numpy.array(
    proj_3D.get_schematic().get_result_item('Tasks\\Tran1\\TD Signals\\O2,3',  ).get_data())



out_pulse_duration = []
# arr_intersected_pts =[]
for i, df_o23_ in enumerate(recorded_o23):
    eta_OM=( df_o23_[key_interpolated_periodic_avg_square]*2)
    ls = LineString(numpy.array((df_o23_[0],eta_OM.values)).T)
    half_maximum = eta_OM[trusted_signal(df_o23_)].max()/2
    ls2 = LineString(numpy.array([[min(df_o23[0]),half_maximum],
                                  [max(df_o23[0]),half_maximum]]))
    intersected_pts= [pt for pt in ls.intersection(ls2).geoms]
    intersected_pts.sort(key = lambda pt:pt.x)
    possible_distance = [ intersected_pts[i].distance(intersected_pts[i+1]) for i in range(len(intersected_pts)-1)    ]
    out_pulse_duration.append(max(possible_distance))
    # out_pulse_duration.append(intersected_pts[-1].distance(intersected_pts[-2]))
    # shapely.lib.intersection_all([ls,ls2])
    # arr_intersected_pts.append(intersected_pts)

out_pulse_duration = numpy.array(out_pulse_duration)

(eta_inf, tau, Dt_right),cov = curve_fit(   _func_eta_OM_peak, Dt_lefts, eta_OM_max ,p0= [0.88,3e-9,0.5e-9])
fig,axs = plt.subplots(4,1 ,figsize=(4,6),constrained_layout = True,sharex= True)
axs[0].plot(1e9*Dt_lefts,out_pulse_duration,'.',label = "output pulse duration" )
axs[0].set_ylabel ("duration (ns)")
_func_out_pulse_duration = lambda Dt_left,a,Dt_OM:a * Dt_left+Dt_OM
__filter= Dt_lefts>1e-9
(a,Dt_OM2,),cov = curve_fit(_func_out_pulse_duration, Dt_lefts[__filter]*1e9, out_pulse_duration[__filter])
__Dt_lefts = numpy.linspace(-0.e-9, max(Dt_lefts),1000)
axs[0].plot(1e9*__Dt_lefts, _func_out_pulse_duration( __Dt_lefts*1e9,a, Dt_OM2),'--',label = r"$%.2f (\Delta t + %.2f~\rm ns)$"%(a,Dt_OM2 /a))
axs[1].plot(1e9*Dt_lefts, eta_OM_max,'.',label = "$\eta_{OM, peak}$")
axs[1].plot(1e9*__Dt_lefts, _func_eta_OM_peak(__Dt_lefts,eta_inf, tau, Dt_right),'--',label = r"$%.3f [1-exp(-\frac{t+ %.2f\ \rm{ns}}{%.2f\ \rm{ns}})]^2$"%(eta_inf, Dt_right*1e9, tau*1e9,))
# axs[0].plot(df_test_01[0],df_test_01[key_interpolated_periodic_avg_square]*2 ,label = 'convolve')

(P23_inf, tau2, Dt_right2) = [0.7254, 1e9 * tau, -0.9]
# (P23, tau2, Dt_right2),cov = curve_fit(   _func_eta_OM_peak, df_test_01[0],df_test_01[key_interpolated_periodic_avg_square]*2,p0= (P23, tau2, Dt_right2) )
logger.info((P23_inf, tau2, Dt_right2))
# axs[0].plot(df_test_01[0], _func_eta_OM_peak(df_test_01[0].values, P23_inf, tau2, Dt_right2), label ='convolve, fitted')
_func_eta_OM_peak2 = lambda  t, Dt: numpy.interp((t +Dt)*1e9,df_test_01[0],2*df_test_01[key_interpolated_periodic_avg_square] )
(Dt_OM , ),cov =  curve_fit(_func_eta_OM_peak2,     Dt_lefts, eta_OM_max,p0 = [1e-9])
# Dt_OM = 1e-9
# axs[0].plot(1e9*Dt_lefts,_func_eta_OM_peak2(Dt_lefts,Dt_OM),label = "Dt_OM = %.2f ns"%(1e9*Dt_OM))
# axs[0].plot(df['time/ns'], df['signal']**2)

axs[2].plot(1e9*__Dt_lefts, func_G_cav_peak(__Dt_lefts),label = '$G_{cav,peak}$')

axs[3].plot(1e9*Dt_lefts, eta_OM_max*G_cav_peak,'.-',label  = "$G_{OM,peak}$")
# axs[3].plot(1e9*__Dt_lefts, _func_eta_OM_peak(__Dt_lefts,eta_inf, tau, Dt_right)*func_G_cav_peak(__Dt_lefts),'.-',label  = "$G_{OM,peak}$")

for ax in axs:ax.legend()
plt.xlabel("$\Delta t$ (ns)")
plt.savefig("G_vs_Delta_t.eps")




