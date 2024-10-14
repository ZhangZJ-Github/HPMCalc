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
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import curve_fit
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
proj: cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch01.1.cst",
                                                        allow_interactive=True)
o23 = numpy.array(proj.get_3d().get_result_item(r'1D Results\Port signals\o2,3[1.0,0.0,signal1],[f_target]').get_data())
o33 = numpy.array(proj.get_3d().get_result_item(r'1D Results\Port signals\o3,3[1.0,0.0,signal1],[f_target]').get_data())
i3 = numpy.array(proj.get_3d().get_result_item(r'1D Results\Port signals\i3[1.0,0.0,signal1],[9.3]').get_data())

f0 = 9.3e9
omega_0 = 2*numpy.pi *f0

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
def to_df(o23):
    df_o23 = pandas.DataFrame(o23)
    df_o23[key_period_for_calculate_avg] = df_o23[0] * 1e-9 // (5 / f0)
    df_o23[key_square] = df_o23[1] ** 2
    df_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
    df_o23['interpolated_periodic_avg_square'] = numpy.interp(df_o23[0], df_time_avg[0], df_time_avg['square'],)
    df_o23[key_complex] = scipy.signal.hilbert(df_o23[1])
    return df_o23


df_o23 = to_df(o23)
df_o33 = to_df(o33)
df_i3 = to_df(i3)
df_o23_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
trusted_idx=  numpy.where((df_o23[0]>3 )&(df_o23[0]<38))[0]
i3_interpolator = interp1d(df_i3[0], df_i3[key_complex], bounds_error=False, fill_value=0  # "extrapolate"
                           )

o23_interpolator = interp1d(df_o23[0], df_o23[key_complex], bounds_error=False, fill_value=0  # "extrapolate"
                            )
o33_interpolator = interp1d(df_o33[0], df_o33[key_complex], bounds_error=False, fill_value=0  # "extrapolate"
                            )

def _o23_abs(t, Q, o23_inf, Dt):
    return    numpy.piecewise(t,[t<=Dt,t>Dt],[0,lambda t:o23_inf*(1-numpy.exp(-omega_0*(t-Dt) / (2* Q)))])
o23_inf = (2*df_o23_time_avg[key_square].values[-1])**0.5
__o23_abs = lambda t,Q,Dt_23: _o23_abs(t,Q,o23_inf,Dt_23)
(Q, Dt_23),cov = (Q, Dt_23),cov = curve_fit(__o23_abs, df_o23[0][trusted_idx].values*1e-9, numpy.abs(df_o23[key_complex])[trusted_idx],p0= (200,1e-9)
                           )

plt.figure()
plt.plot(df_o23[0], numpy.abs(df_o23[key_complex]),label = 'raw data')
# plt.plot(df_o23[0][trusted_idx], numpy.abs(df_o23[key_complex])[trusted_idx],label = 'data for fitting')
plt.plot(df_o23[0],__o23_abs(df_o23[0].values*1e-9,Q,Dt_23),label = 'fitted: Q = %.2f, Dt = %.2f ns'%(Q,Dt_23*1e9)
         )
plt.xlabel("time / ns")
plt.ylabel("normalized output signal")
plt.legend()

def func_o33_abs(t,i3_interpolator,o23_interpolator,n34):
    return numpy.abs( i3_interpolator(t)) - n34*numpy.abs(o23_interpolator(t))
_func_o33_abs = lambda t, n34 :func_o33_abs(t, i3_interpolator,o23_interpolator, n34)
(n34 ,),cov = curve_fit(_func_o33_abs, df_o33[0], numpy.abs(df_o33[key_complex]))

plt.figure()
plt.plot( df_o33[0],numpy.abs(df_o33[key_complex]),label = 'o33, original')
plt.plot(df_o33[0], _func_o33_abs(df_o33[0], n34, ), label ='fitted')
plt.legend()


def func_d_o23_abs_square_dt(t, i3_interpolator, o23_interpolator,o33_interpolator,k,g):
    return k*(numpy.abs(i3_interpolator(t))**2-numpy.abs(o33_interpolator(t))**2 -g* numpy.abs(o23_interpolator(t))**2)
t_end = 0.9*df_o23[0].max()
# g =( numpy.abs(i3_interpolator(t_end))**2 - numpy.abs(o23_interpolator(t_end))**2 -numpy.abs( o33_interpolator(t_end))**2) / numpy.abs(o23_interpolator(t_end))**2+1
_func_d_o23_abs_square_dt =lambda t, k,g: func_d_o23_abs_square_dt(t, i3_interpolator, o23_interpolator,o33_interpolator, k, g)

(k,g),cov = curve_fit(_func_d_o23_abs_square_dt, df_o23_time_avg[0].values[1:][5:],(2*numpy.diff(df_o23_time_avg[key_square])/ numpy.diff(df_o23_time_avg[0]))[5:])
# k,g = 0.4, 1.02
plt.figure()
plt.plot(df_o23_time_avg[0][1:],2*numpy.diff(df_o23_time_avg[key_square])/ numpy.diff(df_o23_time_avg[0]),'.-',label = r'$\frac{d |o_{23}(t)|^2}{dt}$')
plt.plot(df_o23[0],_func_d_o23_abs_square_dt(df_o23[0],k,g),label = 'fitted')
plt.legend()

t = numpy.linspace(0, 40, 200)
EPS = 1e-6


i3_abs = 1

def _func_d_o23_abs_square_dt(o23_abs_square,t, i3_interpolator,k,g):
    return k*(numpy.abs(i3_interpolator(t))**2- (numpy.abs( i3_interpolator(t)) - n34* o23_abs_square**0.5 )**2-g* o23_abs_square)


plt.figure()
plt.plot(df_o23[0], numpy.abs(df_o23[key_complex])**2,label = '$|o23|^2$, original')
for n34 in [n34]:#numpy.linspace(0.1,0.9,9):
    sol = odeint( _func_d_o23_abs_square_dt ,
                                    EPS, t,args=(lambda t:i3_abs,k,g))
    plt.plot(t, ( sol) ,label = 'n34 = %.1f'%n34)
plt.legend()



Dt_left_33 = 2
ts = numpy.array([numpy.linspace( i *Dt_left_33, (i+1) *Dt_left_33 ,100) for i in range(int(40 // Dt_left_33))])
sols = [EPS,EPS]
o3_abs_data = [[-1.1*Dt_left_33, 0],[i3_abs,i3_abs]]#0行：时间，1行：信号值

func_o3 = lambda t: numpy.interp(t, o3_abs_data[0], o3_abs_data[1])
func_i3 = lambda t: func_o3(t - Dt_left_33)
for _ts in ts:
    sols =numpy.hstack([sols, odeint( _func_d_o23_abs_square_dt ,
                                sols[-1], _ts,args=(func_i3,k,g))[:,0]])
    o3_abs_data[1] =numpy.hstack ((o3_abs_data[1],func_o33_abs(_ts, func_i3, lambda t: numpy.interp(t,_ts,sols[-len(_ts):]**0.5),n34)))
    o3_abs_data[0] =numpy.hstack([o3_abs_data[0], _ts])


fig, axs=  plt.subplots(2,1, sharex = True)
axs[0].plot(o3_abs_data[0],(o3_abs_data[1])**2,label = '$|o_{33}(t)|^2$')
axs[1] .plot(o3_abs_data[0],sols,label = '$|o_{23}(t)|^2$')
for ax in axs:ax.legend()

