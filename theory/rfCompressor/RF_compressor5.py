# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 13:07
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : RF_compressor5.py
# @Software: PyCharm
import cst.results,scipy
import skrf
import numpy, matplotlib,pandas,scipy
import cst.results
matplotlib.use('tkagg')
import matplotlib .pyplot as plt
from scipy.optimize import curve_fit
plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_target= 9.3e9
def to_df(o23):
    df_o23 = pandas.DataFrame(o23)
    df_o23[key_period_for_calculate_avg] = df_o23[0] * 1e-9 // (2 / f_target)
    df_o23[key_square] = df_o23[1] ** 2
    df_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
    df_o23[key_interpolated_periodic_avg_square] = numpy.interp(df_o23[0], df_time_avg[0], df_time_avg['square'],)
    df_o23[key_complex] = scipy.signal.hilbert(df_o23[1])
    return df_o23
# nw = skrf.network.Network(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch01.1.s2p")
# vf = skrf.VectorFitting(nw)
# vf.auto_fit()
# plt.figure()
# ax =vf.plot_s_mag()

# 需包含放能阶段的S参数
proj_3D :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\rfCompressor\test_SES_switch01.1 - copy.cst",
                                                        allow_interactive=True)
proj_3D_charging :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\rfCompressor\test_SES_switch01.1-energy_storage.cst",
                                                        allow_interactive=True)

# proj_discharging :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch02.cst",
#                                                         allow_interactive=True)
# proj_3D_charging :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch02.1.cst",
#                                                         allow_interactive=True)
S33_3D  =numpy.array(proj_3D.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3',).get_data())
S23_3D  =numpy.array(proj_3D.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3',).get_data())
Zref_2 = numpy.array(proj_3D.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 2(1)').get_data())
Zref_3 = numpy.array(proj_3D.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 3(1)').get_data())
Zref_3_interpolator = scipy.interpolate.interp1d(numpy.real(Zref_3[:,0]),numpy.real(Zref_3[:, 1]),fill_value="extrapolate")
Zref_2_interpolator = scipy.interpolate.interp1d(numpy.real(Zref_2[:,0]),numpy.real(Zref_2[:, 1]),fill_value="extrapolate")

S33_3D_interpolator =lambda f: numpy.piecewise(f,[numpy.real(f)>=0, ],
                           [lambda f:numpy.interp(numpy.real(f),numpy.real(S33_3D[:,0]),S33_3D[:, 1],left = complex(0),right = complex(0)),
                           lambda f:numpy.interp(-numpy.real(f),numpy.real(S33_3D[:,0]),     numpy.conj(S33_3D[:, 1]),left = complex(0),right = complex(0))
                                                                ],)
S23_3D_interpolator =lambda f: numpy.piecewise(f,[numpy.real(f)>=0,  ],
                           [lambda f:numpy.interp(numpy.real(f),numpy.real(S23_3D[:,0]),S23_3D[:, 1],left = 0,right = 0),
                           lambda f:numpy.interp(-numpy.real(f),numpy.real(S23_3D[:,0]),     numpy.conj(S23_3D[:, 1]),left = 0,right = 0)
                                                                ])



S33_3D_charging  =numpy.array(proj_3D_charging.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3',).get_data())
S23_3D_charging   =numpy.array(proj_3D_charging.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3',).get_data())
Zref_2_charging  = numpy.array(proj_3D_charging.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 2(1)').get_data())
Zref_3_charging  = numpy.array(proj_3D_charging.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 3(1)').get_data())
Zref_3_interpolator_charging  = scipy.interpolate.interp1d(numpy.real(Zref_3_charging [:,0]),numpy.real(Zref_3_charging [:, 1]),fill_value="extrapolate")
Zref_2_interpolator_charging  = scipy.interpolate.interp1d(numpy.real(Zref_2_charging [:,0]),numpy.real(Zref_2_charging [:, 1]),fill_value="extrapolate")

S33_3D_interpolator_charging  =lambda f: numpy.piecewise(f,[numpy.real(f)>=0, ],
                           [lambda f:numpy.interp(numpy.real(f),numpy.real(S33_3D_charging [:,0]),S33_3D_charging [:, 1],left = complex(0),right = complex(0)),
                           lambda f:numpy.interp(-numpy.real(f),numpy.real(S33_3D_charging [:,0]),     numpy.conj(S33_3D_charging [:, 1]),left = complex(0),right = complex(0))
                                                                ],)
S23_3D_interpolator_charging  =lambda f: numpy.piecewise(f,[numpy.real(f)>=0,  ],
                           [lambda f:numpy.interp(numpy.real(f),numpy.real(S23_3D_charging [:,0]),S23_3D_charging [:, 1],left = 0,right = 0),
                           lambda f:numpy.interp(-numpy.real(f),numpy.real(S23_3D_charging [:,0]),     numpy.conj(S23_3D_charging [:, 1]),left = 0,right = 0)
                                                                ])


f0 = 9.3e9
from scipy.fft import fft,fftfreq,ifft
t = numpy.linspace(-10e-9, 40e-9, 100000 )
i3 =numpy.piecewise(t, [(t>0)&(t<10e-9),], [lambda t:numpy.sin(2*numpy.pi*f0*t) ,0])
i3_fft = fft(i3)


freqs = fftfreq(len(t),t[1]-t[0])
plt.plot(freqs,i3_fft,label= '$F(i_3(t))$')
plt.plot(freqs, numpy.abs(S23_3D_interpolator( freqs.astype(complex)/1e9)),label = '$S_{23}$')
o23_fft = S23_3D_interpolator( freqs.astype(complex)/1e9) *i3_fft
o33_fft = S33_3D_interpolator( freqs.astype(complex)/1e9) *(i3_fft+(100.0+0j))
plt.plot(freqs,o23_fft,label = '$S_{23}I_3$' )


plt.legend()
o23_fft_ifft = ifft(o23_fft)
o33_fft_ifft = ifft(o33_fft)
plt.figure()
plt.plot(t, o23_fft_ifft, label = '$F^{-1}(S_{23} I_3)$')
plt.plot(t, o33_fft_ifft, label = '$F^{-1}(S_{33} I_3)$')
plt.plot(t,i3,label = '$i_3$')
plt.legend()



# nw_extrapolate_to_dc= nw.extrapolate_to_dc(dc_sparam=[[0,1],[1,0]])
# nw.impulse_response()
# plt.figure()
# plt.plot( nw_extrapolate_to_dc.f, nw_extrapolate_to_dc.s[:,0])


impulse_response_23 =ifft(S23_3D_interpolator(freqs.astype(complex)/1e9)) [:len(t) //2]
impulse_response_33 =ifft(S33_3D_interpolator(freqs.astype(complex)/1e9))[:len(t) //2]

impulse_response_23_charging =ifft(S23_3D_interpolator_charging(freqs.astype(complex)/1e9)) [:len(t) //2]
impulse_response_33_charging =ifft(S33_3D_interpolator_charging(freqs.astype(complex)/1e9)) [:len(t) //2]
plt.figure()
plt.plot(t[:len(t) //2]-t[0], impulse_response_23)
plt.plot(t[:len(t) //2]-t[0], impulse_response_33)


o23_from_convolve =  scipy.signal.convolve(i3,impulse_response_23,)
o33_from_convolve =  scipy.signal.convolve(i3,impulse_response_33,)
o23_from_convolve_charging =  scipy.signal.convolve(i3,impulse_response_23_charging,)
o33_from_convolve_charging =  scipy.signal.convolve(i3,impulse_response_33_charging,)
plt.figure()
plt.plot(t, i3,label = 'i3')
plt.plot(numpy.linspace(t[0],t[0]+(t[1]-t[0])*len(o23_from_convolve),len(o23_from_convolve)), o23_from_convolve,label = 'o23')
plt.plot(numpy.linspace(t[0],t[0]+(t[1]-t[0])*len(o33_from_convolve),len(o33_from_convolve)), o33_from_convolve,label = 'o33')
plt.plot(numpy.linspace(t[0],t[0]+(t[1]-t[0])*len(o23_from_convolve_charging),len(o23_from_convolve_charging)), o23_from_convolve_charging,label = 'o23, charging')
plt.plot(numpy.linspace(t[0],t[0]+(t[1]-t[0])*len(o33_from_convolve_charging),len(o33_from_convolve_charging)), o33_from_convolve_charging,label = 'o33, charging')
plt.legend()

Dt_left = 0.5e-9
dt = t[1]-t[0]
t= numpy.arange(0-10e-9, 0+Dt_left, dt)
i3_charging =numpy.piecewise(t, [(t>-10e-9)&(t<0),], [lambda t:numpy.sin(2*numpy.pi*f0*t) ,0])
i3_discharging =numpy.piecewise(t, [(t>0),], [lambda t:numpy.sin(2*numpy.pi*f0*(t)) ,0])

# plt.figure()
o33_from_convolve_charging = scipy.signal.convolve(i3_charging, impulse_response_33_charging, )
o33_from_convolve_discharging = scipy.signal.convolve(i3_discharging, impulse_response_33, )
o33= o33_from_convolve_charging+o33_from_convolve_discharging

while t[-1]<40e-9:

    # i3_charging = numpy.hstack([i3_charging,o33_from_convolve_charging[:len(t)][-int(Dt_MESS//dt):]])
    i3_discharging = numpy.hstack([i3_discharging,o33[:len(t)][-int(Dt_left//dt):]])
    t = numpy.arange(t[0], t[-1]+Dt_left, dt)
    o33 = scipy.signal.convolve(i3_discharging, impulse_response_33, )

o23 = scipy.signal.convolve(i3_discharging, impulse_response_23, )
plt.figure()
# plt.plot(t,i3_charging)
plt.plot(t,i3_discharging,label = 'i3')
plt.plot(t, o33[:len(t)],label ='o33')
plt.plot(t, o23[:len(t)],label ='o23')
plt.legend()

df_i3_discharging=to_df(numpy.vstack((t*1e9, i3_discharging)).T.astype(float))
df_o33=to_df(numpy.vstack((t*1e9, o33[:len(t)])).T.astype(float))
df_o23=to_df(numpy.vstack((t*1e9, o23[:len(t)])).T.astype(float))
plt.figure()
plt.plot(df_i3_discharging[0],numpy.abs(df_i3_discharging[key_complex])**2,label = 'i3')
plt.plot(df_i3_discharging[0],numpy.abs(df_o33[key_complex])**2,label = 'o33')
plt.plot(df_i3_discharging[0],numpy.abs(df_o23[key_complex])**2,label = 'o23',lw = 5)
plt.legend()

plt.figure()
plt.plot(df_i3_discharging[0], (df_i3_discharging[key_interpolated_periodic_avg_square]) * 2,label = 'i3')
plt.plot(df_i3_discharging[0],(df_o33[key_interpolated_periodic_avg_square])*2,label = 'o33')
plt.plot(df_i3_discharging[0],(df_o23[key_interpolated_periodic_avg_square])*2,label = 'o23',lw = 5)
plt.legend()


# plt.plot(t,  scipy.signal.convolve(i3,impulse_response_23_charging,))

plt.figure()
plt.plot(numpy.linspace(t[0],t[0]+(t[1]-t[0])*len(o33_from_convolve_charging),len(o33_from_convolve_charging)), o33_from_convolve_charging,label = 'o33, charging when switch is off')
plt.plot(numpy.linspace(t[0],t[0]+(t[1]-t[0])*len(o33_from_convolve_discharging),len(o33_from_convolve_discharging)), o33_from_convolve_discharging,label = 'o33, charging when switch is on')
plt.plot(numpy.linspace(t[0],t[0]+(t[1]-t[0])*len(o33_from_convolve_discharging),len(o33_from_convolve_discharging)), o33_from_convolve_charging+o33_from_convolve_discharging,label = 'o33, discharging when switch is on')


plt.legend()