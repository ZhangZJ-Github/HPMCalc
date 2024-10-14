# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 13:19
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : RF_compressor.py
# @Software: PyCharm
import cst.results
import matplotlib
import numpy
import pandas
import scipy.interpolate

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
proj: cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch01.1.cst",
                                                        allow_interactive=True)
o23 = numpy.array(proj.get_3d().get_result_item(r'1D Results\Port signals\o2,3[1,0,signal1],[9.3]').get_data())
o33 = numpy.array(proj.get_3d().get_result_item(r'1D Results\Port signals\o3,3[1,0,signal1],[9.3]').get_data())
i3 = numpy.array(proj.get_3d().get_result_item(r'1D Results\Port signals\i3[1.0,0.0,signal1],[9.3]').get_data())
e_switchcav = numpy.array(proj.get_3d().get_result_item(
    '1D Results\\Probes\\E-Field\\Probe Signals\\E-Field (9 30 280)(X) [3[1,0,signal1],[9.3]]', ).get_data())
# e_switchcav[:,1]/= e_switchcav[:,1].max()+0.1
plt.figure()
plt.plot(*i3.T, label='i3')
plt.plot(*o23.T, label='o23')
plt.plot(*o33.T, label='o33')
# plt.plot(*e_switchcav.T,label = 'e_switchcav')
plt.legend()
f0 = 9.304e9
key_period_for_calculate_avg = 'period_for_calculate_avg'
e_switchcav_square = pandas.DataFrame(data=e_switchcav)
e_switchcav_square[1] = e_switchcav_square[1] ** 2
e_switchcav_square[key_period_for_calculate_avg] = e_switchcav_square[0] * 1e-9 // (1 / f0)
e_switchcav_square_periodic_avg = e_switchcav_square.groupby(key_period_for_calculate_avg).mean()


def v(t, v_inf, omega, Q, Delta_t):
    # 和电压成正比的量
    return v_inf * (1 - numpy.exp(-omega / (2 * Q) * (t - Delta_t)))


from scipy.optimize import curve_fit

v_ = lambda t, v_inf, Q, Delta_t: v(t, v_inf, 2 * numpy.pi * f0, Q, Delta_t)
params, cov = curve_fit(v_, e_switchcav_square_periodic_avg[0].values * 1e-9,
                        e_switchcav_square_periodic_avg[1].values ** 0.5,
                        p0=[e_switchcav_square_periodic_avg[0].values[-2] ** 0.5, 200, 10e-9]
                        )
_, Q, Delta_t = params
plt.figure()
plt.plot(*e_switchcav_square_periodic_avg.values.T, label='original signal')
plt.plot(e_switchcav_square_periodic_avg[0].values, v_(e_switchcav_square_periodic_avg[0].values * 1e-9, *params) ** 2,
         label='Q = %.0f, $\Delta t$ = %.1f ns' % (Q, Delta_t * 1e9))
plt.legend()

key_complex = 'complex'


def to_df(o23):
    df_o23 = pandas.DataFrame(o23)
    df_o23[key_period_for_calculate_avg] = df_o23[0] * 1e-9 // (20 / f0)
    df_o23['square'] = df_o23[1] ** 2
    df_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
    df_o23['interpolated_periodic_avg_square'] = numpy.interp(df_o23[0], df_time_avg[0], df_time_avg['square'],)
    df_o23[key_complex] = scipy.signal.hilbert(df_o23[1])
    return df_o23


df_o23 = to_df(o23)
df_o33 = to_df(o33)
df_e_switch = to_df(e_switchcav)
df_i3 = to_df(i3)
plt.figure()
plt.plot(df_o23[0], df_o23[1], label='original')
plt.plot(df_o23[0], df_o23['complex'], label='complex')
plt.plot(df_o23[0], numpy.abs(df_o23['complex']), label='amplitude')
plt.legend()

Nsamples = len(df_i3)


# M = numpy.zeros((2 * Nsamples, 4), dtype=complex)
# M[:Nsamples, 0] = M[Nsamples:, 2] = df_i3[key_complex]
# M[:Nsamples, 1] = M[Nsamples:, 3] = df_e_switch[key_complex]
# M = numpy.matrix(M)
# S23, S24, S33, S34 = numpy.linalg.pinv(M) * numpy.matrix([[*df_o23[key_complex], *df_o33[key_complex]]]).T
# plt.figure()
# plt.plot(df_o23[0], df_o23[1], label='o23, original')
# plt.plot(df_o23[0], S23[0, 0] * df_i3[key_complex] + S24[0, 0] * df_e_switch[key_complex], label='o23, from S matrix')
#
# plt.plot(df_o33[0], df_o33[1], label='o33, original')
# plt.plot(df_o33[0], S33[0, 0] * df_i3[key_complex] + S34[0, 0] * df_e_switch[key_complex], label='o33, from S matrix')
# plt.legend()


def o23_(t, i3_interpolator, e4_interpolator, S23, S24, Dt_i3, Dt_e4):
    # i4 = e4_interpolator(t)  /(1+m)
    # o4 = i4*m
    return S23 * numpy.abs(i3_interpolator(t - Dt_i3)) * numpy.exp(
        1j * numpy.angle(i3_interpolator(t))) + S24 * numpy.abs(e4_interpolator(t - Dt_e4)) * numpy.exp(
        1j * numpy.angle(e4_interpolator(t)))
    # return S23 * i3_interpolator(t - Dt_i3) + S24 * e4_interpolator(t - Dt_e4)


from scipy.interpolate import interp1d

i3_interpolator = interp1d(df_i3[0], df_i3[key_complex], bounds_error=False, fill_value=0  # "extrapolate"
                           )
e_switchcav_interpolator = interp1d(df_e_switch[0], df_e_switch[key_complex], bounds_error=False, fill_value=0
                                    # "extrapolate"
                                    )
o23_interpolator = interp1d(df_o23[0], df_o23[key_complex], bounds_error=False, fill_value=0  # "extrapolate"
                            )
o33_interpolator = interp1d(df_o33[0], df_o33[key_complex], bounds_error=False, fill_value=0  # "extrapolate"
                            )
o23__ = lambda t, S23_real, S23_imag, S24_real, S24_imag, Dt_i3, Dt_e4: numpy.real(
    o23_(t, i3_interpolator, e_switchcav_interpolator, S23_real + 1j * S23_imag, S24_real + 1j * S24_imag, Dt_i3,
         Dt_e4))
(S23_real, S23_imag, S24_real, S24_imag, Dt_o23_i3, Dt_o23_e4), cov = curve_fit(o23__, df_o23[0],
                                                                                df_o23[1])
(S33_real, S33_imag, S34_real, S34_imag, Dt_o33_i3, Dt_o33_e4), cov = curve_fit(o23__, df_o33[0],
                                                                                df_o33[1])


def _func1( t, m_real,m_imag, Dt_43):
    m = m_real+1j*m_imag
    return numpy.real((1+m)**2 / m * (S34_real+1j*S34_imag)*i3_interpolator(t))
(m_real,m_imag, ), cov = curve_fit(_func1, df_e_switch[0],     df_e_switch[1])
m = m_real+1j*m_imag

plt.figure()
plt.plot(df_o23[0], df_o23[1], label='o23, original')
plt.plot(df_o23[0], o23__(df_o23[0], S23_real, S23_imag, S24_real, S24_imag, Dt_o23_i3, Dt_o23_e4),
         label='o23, S matrix and delay')
plt.plot(df_o33[0], df_o33[1], label='o33, original')
plt.plot(df_o33[0], o23__(df_o33[0], S33_real, S33_imag, S34_real, S34_imag, Dt_o33_i3, Dt_o33_e4),
         label='o33, S matrix and delay')
plt.legend()

t_end = max(df_o23[0]) * 0.9
g_over_k = ((lambda t: (
    (numpy.abs(i3_interpolator(t)) ** 2 - numpy.abs(o23_interpolator(t)) ** 2 - numpy.abs(o33_interpolator(t)) ** 2))))(
    t_end) / numpy.abs(e_switchcav_interpolator(t_end)) ** 2


def __de42_dt(t, k, ):
    g = g_over_k * k
    return k * (numpy.abs(i3_interpolator(t)) ** 2 - numpy.abs(o23_interpolator(t)) ** 2 - numpy.abs(
        o33_interpolator(t)) ** 2) - g * numpy.abs(e_switchcav_interpolator(t)) ** 2


idx = numpy.where((df_e_switch[0] > 2.5) & (df_e_switch[0] < 150))

(k,), cov = curve_fit(__de42_dt, df_e_switch[0][idx[0]].values[1:],
                      numpy.diff(2*df_e_switch['interpolated_periodic_avg_square'][idx[0]]) / numpy.diff(
                          df_e_switch[0][idx[0]]))
plt.figure()
# plt.plot(df_e_switch[0][1:], numpy.diff(numpy.abs(df_e_switch[key_complex])**2)/numpy.diff(df_e_switch[0]))
# plt.plot(df_e_switch[0],(numpy.abs(df_e_switch[key_complex])**2))
# plt.plot(df_e_switch[0][1:], numpy.diff(numpy.abs(scipy.signal.hilbert(df_e_switch[1]**2)))/numpy.diff(df_e_switch[0]))
plt.plot(df_e_switch[0][idx[0]].values[1:],
         numpy.diff(2*df_e_switch['interpolated_periodic_avg_square'][idx[0]]) / numpy.diff(df_e_switch[0][idx[0]]),
         label=r'$\frac{d |e_4(t)|^2}{dt}$')
# k= 22434888.339331454#, 0.0055548065875803865
# g = k*g_over_k
plt.plot(df_o23[0], __de42_dt(df_o23[0], k, ), label='fitted')
plt.legend()
#
# proj_left: cst.results.ProjectFile = cst.results.ProjectFile(
#     r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch01.1.left.cst",
#     allow_interactive=True)
# df_left_i3 = to_df(
#     numpy.array(proj_left.get_3d().get_result_item(r'1D Results\Port signals\i3[1.0,0.0,signal1],[9.304]').get_data()))
# df_left_o33 = to_df(
#     numpy.array(proj_left.get_3d().get_result_item('1D Results\\Port signals\\o3,3[1,0,signal1],[9.304]').get_data()))
# df_left_o13 = to_df(
#     numpy.array(proj_left.get_3d().get_result_item('1D Results\\Port signals\\o1,3[1,0,signal1],[9.304]').get_data()))
#
# df_left_i3_interpolator = interp1d(df_left_i3[0], df_left_i3[key_complex], bounds_error=False, fill_value=0)
# func_o_left_33__ = lambda t, S_left_33_real, S_left_33_imag, Dt_left_33: numpy.real(
#     (S_left_33_real + 1j * S_left_33_imag) * numpy.abs(df_left_i3_interpolator(t - Dt_left_33)) * numpy.exp(
#     1j * numpy.angle(df_left_i3_interpolator(t))))
# Nsamples_left = len( df_left_o33[0])
# (S_left_33_real, S_left_33_imag, Dt_left_33), cov = curve_fit(func_o_left_33__, df_left_o33[0][:Nsamples_left],
#                                                               numpy.real(df_left_o33[1][:Nsamples_left]), )
# Dt_left_33 = 2
# plt.figure()
# plt.plot(df_left_o33[0], df_left_o33[1], label='o_left_33, original')
# plt.plot(df_left_o33[0], func_o_left_33__(df_left_o33[0], S_left_33_real, S_left_33_imag, Dt_left_33),
#          label='o_left_33, fitted')
# # func2_o_left_33__ = lambda t,  Dt_left_33: numpy.real(
# #     (S_left_33_real + 1j * S_left_33_imag) * numpy.abs(df_left_i3_interpolator(t - Dt_left_33)) * numpy.exp(
# #     1j * numpy.angle(df_left_i3_interpolator(t))))
# # (Dt_left_33,), cov = curve_fit(func2_o_left_33__, df_left_o33[0],                                                              numpy.real(df_left_o33[1]), )
# #
# # plt.plot(df_left_o33[0], func_o_left_33__(df_left_o33[0], S_left_33_real, S_left_33_imag, Dt_left_33),
# #          label='o_left_33, fitted2')
#
# plt.legend()




def func_right_P_3_out (t, i3_interpolator ,e4_interpolator , K33, K34):
    i3 = i3_interpolator(t)
    e4 = e4_interpolator(t)
    return K33*numpy.abs(i3)**2 + K34*numpy.abs(e4) **2+2*K33*K34*numpy.abs(i3*e4)
func_right_P_3_out_  = lambda t, K33,K34 :func_right_P_3_out(t, i3_interpolator, e_switchcav_interpolator,K33,K34)
(K33,K34),cov = curve_fit(func_right_P_3_out_ ,df_o33[0][idx[0]],numpy.abs(df_o33[key_complex][idx[0]])**2,)
plt.figure()
plt.plot(df_o33[0],numpy.abs(df_o33[key_complex])**2,label = 'P33, original')
# plt.plot(df_o33[0][idx[0]],numpy.abs(df_o33[key_complex][idx[0]])**2,label = 'P33, original2')
plt.plot(df_o33[0],func_right_P_3_out_(df_o33[0], K33,K34),label = 'P33, fitted')
plt.legend()



def  func_right_o33(t, i3_interpolator, e4_interpolator , K33,k34):
    return K33*numpy.abs(i3_interpolator(t)) - k34*numpy.abs( e4_interpolator(t))

func_right_o33_  = lambda t, n43 :func_right_o33(t, i3_interpolator, e_switchcav_interpolator,1,1/n43)
func_right_o23_  = lambda t, n42 :func_right_o33(t, i3_interpolator, e_switchcav_interpolator,0,-1/n42)
(n43),cov = curve_fit(func_right_o33_ ,df_o33[0][idx[0]],numpy.abs(df_o33[key_complex][idx[0]]),)
(n42),cov = curve_fit(func_right_o23_ ,df_o23[0][idx[0]],numpy.abs(df_o23[key_complex][idx[0]]),)
plt.figure()
plt.plot(df_o33[0],numpy.abs(df_o33[key_complex]),label = 'o33, original')
plt.plot(df_o33[0],func_right_o33_(df_o33[0],n43),label = 'o33, fitted')
plt.plot(df_o23[0],numpy.abs(df_o23[key_complex]),label = 'o23, original')
plt.plot(df_o23[0],func_right_o23_(df_o23[0], n42),label = 'o23, fitted')
plt.legend()



t = numpy.linspace(0, 200, 2000)
EPS = 0.01
arr_sumP = []
theta_4=  0
i3_abs = 1


def de4_abs_square_dt(e4_abs_square, t, ):
    return  ( k * (i3_abs ** 2 - ( i3_abs -e4_abs_square**0.5 /  n43)**2-(e4_abs_square**0.5 / n42)**2) - k*g_over_k * e4_abs_square)


from scipy.integrate import odeint
sol = odeint(de4_abs_square_dt, EPS, t, )
plt.figure()
plt.plot(t, sol)
plt.plot(df_e_switch[0],numpy.abs( df_e_switch[key_complex])**2)


Dt_left_33 = 1
ts = numpy.array([numpy.linspace( i *Dt_left_33, (i+1) *Dt_left_33 ,10) for i in range(int(3000 // Dt_left_33))])
sols = [EPS,EPS]
o3_abs_data = [[-1.1*Dt_left_33, 0],[i3_abs,i3_abs]]#0行：时间，1行：信号值

def de4_abs_square_dt(e4_abs_square, t, func_i3 ):
    # print(func_i3(t) )
    return  ( k * (func_i3(t) ** 2 -  (func_i3(t) -e4_abs_square**0.5 /  n43)**2-(e4_abs_square**0.5 / n42)**2) - k*g_over_k * e4_abs_square)
func_o3 = lambda t: numpy.interp(t, o3_abs_data[0], o3_abs_data[1])
func_i3 = lambda t: func_o3(t - Dt_left_33)
for _ts in ts:
    sols =numpy.hstack([sols, odeint(de4_abs_square_dt, sols[-1], _ts,args=(func_i3,))[:,0]])
    o3_abs_data[1] =numpy.hstack ((o3_abs_data[1], func_i3(_ts) -(sols[-len(_ts):])**0.5 /  n43))
    o3_abs_data[0] =numpy.hstack([o3_abs_data[0], _ts])


fig, axs=  plt.subplots(3,1, sharex = True)
axs[0].plot(o3_abs_data[0],(o3_abs_data[1])**2)
axs[1].plot(o3_abs_data[0],sols)
axs[2] .plot(o3_abs_data[0],(sols**0.5 / n42)**2 )


