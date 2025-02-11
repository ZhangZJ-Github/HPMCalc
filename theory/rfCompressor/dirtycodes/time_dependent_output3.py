# -*- coding: utf-8 -*-
# @Time    : 2024/11/14 17:54
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : time_dependent_output2.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy

from theory.rfCompressor .time_dependent_output import  *
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch_1-1.paramsweep.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_allCST.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_mode_converter.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_mode_converter.comparison.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_TE10.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_multiway_coupler2.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_15.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.cst"
cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_2_way.lower_Efield_when_ES.cst"
# cst_proj_path =  r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_2_way.lower_Efield_when_ES.cst"
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch.TapperedSwitchCav.2.paramsweep.cst"
# cst_proj_path =         r"E:\CSTprojects\rfCompressor\cascadedHT\SES_allCST.cst"
proj_discharging: cst.results.ProjectFile = cst.results.ProjectFile(cst_proj_path,
                                                                    allow_interactive=True)
proj_charging: cst.results.ProjectFile = cst.results.ProjectFile(
    proj_discharging.filename[:
                              # -len("paramsweep.cst")
                              -len("cst")
    ]
    + "ES.cst",
    allow_interactive=True)
run_id = 0
run_id_charging = 0
gamma_3 = numpy.array(
    proj_charging.get_3d().get_result_item('1D Results\\Port Information\\Gamma\\3(1)', run_id_charging).get_data())
import scipy.constants as C
v_g = C.c ** 2 / (2 * numpy.pi * f_ref / interp1d(gamma_3[:, 0], gamma_3[:, 1], )(f_ref / 1e9).imag)
# gamma_3[:,1] = gamma_3[:,1] .imag*1j# Ignoring attenuation
S33_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S3,3", run_id)
S23_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S2,3", run_id)
S43_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S4,3", run_id)
# S23_discharging [:,1] =S23_discharging [:,1] + S43_discharging[:,1]
# S23_discharging[:, 1] = 1 - S33_discharging[:, 1]
S33_charging = get_S_parameter_from_CST_proj(proj_charging, "S3,3", run_id)
S23_charging = get_S_parameter_from_CST_proj(proj_charging, "S2,3", run_id)
S43_charging = get_S_parameter_from_CST_proj(proj_charging, "S4,3", run_id)
# S23_charging[:, 1] += S43_charging[:, 1]
# S23_charging [:,1] = 1-S33_charging[:,1]
dt = 1 / f_ref / 15.  # 1/f_target / 10
ts = numpy.arange(-200e-9, 0, dt)
# ts = ts[:(len(ts)//2)*2]
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006000000000000015e9)
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006e9)
# resampled_f = numpy.arange(9e9, 9.6e9, 0.0006e9)
# resampled_f = numpy.arange(8.5e9, 10e9, 0.0006e9)
resampled_f = numpy.arange(8.0e9, 10.5e9, 0.0006e9)
# resampled_f = numpy.arange(0e9, 12e9,0.1e9)
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006100000000000015e9)
# compressor.nw_charging = compressor.nw_charging.interpolate(resampled_f)#
# compressor.nw_discharging = compressor.nw_discharging.interpolate(resampled_f)#[band]
compressor = Compressor(
    build_network_from_CST_S_data_1inNout([S33_discharging, S23_discharging, S43_discharging
                                           ]).interpolate(
        resampled_f)
    #   .extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))
    ,
    build_network_from_CST_S_data_1inNout([S33_charging, S23_charging, S43_charging
                                           ]).interpolate(
        resampled_f)
    #  .extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))
    ,
)
ts_for_caching_impulse_response = numpy.arange(-100e-9, 0, dt)
compressor_deembedded = compressor.embed_with_waveguide(gamma_3, -((300 - 40 - 20 * 0)) * 1e-3)
G_cav_deembedded = (1 - numpy.abs(compressor_deembedded.nw_charging.interpolate([f_ref]).s[0, 0, 0]) ** 2) ** -1
# band = "9-9.6GHz"
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006000000000000015e9)
# compressor.nw_charging = compressor.nw_charging.interpolate(resampled_f)#
# compressor.nw_discharging = compressor.nw_discharging.interpolate(resampled_f)#[band]
# compressor_deembedded.nw_charging =compressor_deembedded.nw_charging.interpolate(resampled_f)#[band]
# compressor_deembedded.nw_discharging = compressor_deembedded.nw_discharging.interpolate(resampled_f)#[band]
# compressor.cache_impulse_responses(ts_for_caching_impulse_response)
# compressor_deembedded.cache_impulse_responses(ts_for_caching_impulse_response)
sin = numpy.sin(2 * numpy.pi * f_ref * ts)
# df_i3_charging, df_i3_discharging, df_o33, df_o23 = compressor.run((ts, sin), 10e-9,
#                                                                    compressor.correct_Dt_MESS(
#                                                                        f_ref, 15e-9))
df_i1_charging, df_i1_discharging, df_outs = compressor.run((ts, sin), 150e-9,
                                                            compressor.correct_Dt_MESS(
                                                                f_ref, 100e-9))
df_o33, df_o23,df_o43 = [df_outs[i] for i in range(compressor.nw_charging.nports)]



plt.figure(figsize=(5,4),constrained_layout = True)
ns = 1e-9
plt.plot(df_o23[0]/ns,numpy.abs( df_o23[key_complex])**2+ numpy.abs( df_o43[key_complex])**2,label = r"model 2hole-2, convolved: $\eta_2+\eta_3$",lw = 3)
plt.plot(df_o23[0]/ns, numpy.abs( df_o23[key_complex])**2,label = r"model 2hole-2, convolved: $\eta_2$",#ls = ':'
         )
plt.plot(df_o23[0]/ns,  numpy.abs( df_o43[key_complex])**2,label = "model 2hole-2, convolved: $\eta_3$",#ls = ':'
         )
#
# plt.figure()
#
# for Dt_MESS in numpy.array((*numpy.linspace(0., 1e-9, 5),
#                             *numpy.linspace(1.5e-9, 15e-9, 5),
#                                    # *numpy.linspace(14e-9, 15e-9, 5),
#                             )
#
#                            ):
#     fixed_Dt_MESS = compressor_deembedded.correct_Dt_MESS(
#         f_ref, Dt_MESS)
#     df_i1_charging, df_i1_discharging, df_outs = compressor_deembedded.run((ts, sin), 50e-9,
#                                                                            fixed_Dt_MESS, )
#     df_o33, df_o23,  = [df_outs[i] for i in range(compressor.nw_charging.nports)]
#     index_of_outs = list(range(1, compressor.nw_charging.nports))
#     plt.plot(df_o23[0] / 1e-9,
#              # 1- df_o33[key_complex].abs() ** 2 ,
#              numpy.array([df_outs[i][key_complex].abs() ** 2 for i in index_of_outs]).sum(axis=0),
#              label="Dt_MESS = %.2f ns" % (fixed_Dt_MESS / 1e-9))
#
# plt.legend()

def func(t, tau, t0 , eta_inf):
    return numpy.piecewise(t, [t>t0,],[ lambda t:eta_inf *(1-numpy.exp(-(t-t0)/tau))**2           ,0])

# plt.figure()
from scipy.optimize import curve_fit
trusted_index = (df_o23[0]>0) & (df_o23[0]<2.5e-8)
ts_ = df_o23[0][trusted_index].values

(tau, t0 , eta_inf), cov = curve_fit(func, df_o23[0][trusted_index].values,
                                      ((numpy.abs(df_o23[key_complex])**2 +numpy.abs(df_o43[key_complex])**2)[trusted_index] ).values, p0 = [2.5e-9, 2e-9, 0.762])
plt.plot(ts_/ns, func(ts_, tau, t0 , eta_inf),'--', label = r"fitted: $%.2f \left[1-\exp{\left(-\frac{t-%.2f [ns]} {%.2f [ns]}\right)}\right]^2$"%(eta_inf,  t0/ns, tau/ns))
plt.legend()
plt.xlim(ts_[0]/ns, ts_[-1]/ns)
plt.xlabel('time (ns)')
plt.ylabel('power extraction efficiency')
plt.savefig("transient_PEE_2hole2.pdf")
