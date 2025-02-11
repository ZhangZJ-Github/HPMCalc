# -*- coding: utf-8 -*-
# @Time    : 2024/11/30 11:37
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test2.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy

from theory.rfCompressor.time_dependent_output import *
import scipy.constants as C

# plt.figure(3,1, sharex= True, sharey = True, figsize = (5,6))
plt.figure( figsize = (5,4), #constrained_layout = True
            )
fig = plt.gcf()
axs= [plt.gca()]*3
# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\for_paper\SES_traditional.cst"
cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_traditional.cst"
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
gamma_3_at_f0 = scipy.interpolate.interp1d(gamma_3[:,0].real,gamma_3[:,1])(9.3)
logger.info(gamma_3_at_f0)
v_g = C.c ** 2 / (2 * numpy.pi * f_ref / interp1d(gamma_3[:, 0], gamma_3[:, 1], )(f_ref / 1e9).imag)
# gamma_3[:,1] = gamma_3[:,1] .imag*1j# Ignoring attenuation

S33_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S3,3", run_id)
S23_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S2,3", run_id)
# S43_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S4,3", run_id)
# S23_discharging [:,1] =S23_discharging [:,1] + S43_discharging[:,1]
# S23_discharging[:, 1] = 1 - S33_discharging[:, 1]

S33_charging = get_S_parameter_from_CST_proj(proj_charging, "S3,3", run_id)
S23_charging = get_S_parameter_from_CST_proj(proj_charging, "S2,3", run_id)
# S43_charging = get_S_parameter_from_CST_proj(proj_charging, "S4,3", run_id)
# S23_charging[:, 1] += S43_charging[:, 1]
# S23_charging [:,1] = 1-S33_charging[:,1]

dt = 1 / f_ref / 15.  # 1/f_target / 10
ts = numpy.arange(-20e-9, 0, dt)
# ts = ts[:(len(ts)//2)*2]
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006000000000000015e9)
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006e9)
resampled_f = numpy.arange(8e9, 10.5e9, 0.0006e9)
# resampled_f = numpy.arange(8.5e9, 10e9, 0.0006e9)
# resampled_f = numpy.arange(0e9, 12e9,0.1e9)
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006100000000000015e9)
# compressor.nw_charging = compressor.nw_charging.interpolate(resampled_f)#
# compressor.nw_discharging = compressor.nw_discharging.interpolate(resampled_f)#[band]
compressor = Compressor(
    build_network_from_CST_S_data_1inNout([S33_discharging, S23_discharging, #S43_discharging
                                           ]).interpolate(
        resampled_f)
    #   .extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))
    ,
    build_network_from_CST_S_data_1inNout([S33_charging, S23_charging,# S43_charging
                                           ]).interpolate(
        resampled_f)
    #  .extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))
    ,
)

ts_for_caching_impulse_response = numpy.arange(-400e-9, 0, dt)
L_waveguide=  -1*((300 - 114) ) * 1e-3
compressor_deembedded = compressor.embed_with_waveguide(gamma_3,L_waveguide)
# compressor_deembedded = compressor.embed_with_waveguide(gamma_3, -((300 - 40 - 20 * 0)) * 1e-3)
L = 300e-3
loss_of_ignored_high_order_mode = numpy.exp(-4*gamma_3_at_f0 .real * L )-numpy.abs(compressor.nw_charging.interpolate([f_ref]).s[0, 0, 0]) ** 2 # about -34 dB
G_cav_deembedded = (1 - (numpy.abs(compressor_deembedded.nw_charging.interpolate([f_ref]).s[0, 0, 0]) ** 2+loss_of_ignored_high_order_mode)) ** -1
# G_cav_deembedded = (1 - numpy.abs(compressor.nw_charging.interpolate([f_ref]).s[0, 0, 0]) ** 2 * numpy.exp(-4 *  gamma_3_at_f0.real * L_waveguide)) ** -1
logger.info(G_cav_deembedded)
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
# df_i1_charging, df_i1_discharging, df_outs = compressor.run((ts, sin), 10e-9,
#                                                             compressor.correct_Dt_MESS(
#                                                                 f_ref, 15e-9))
# df_o33, df_o23,  = [df_outs[i] for i in range(compressor.nw_charging.nports)]
# plt.figure()
# plt.plot(df_o23[0], df_o23[1])


i = 0
for Dt_MESS in numpy.array([0.0e-9]

                           ):
    fixed_Dt_MESS = compressor_deembedded.correct_Dt_MESS(
        f_ref, Dt_MESS)
    df_i1_charging, df_i1_discharging, df_outs = compressor_deembedded.run((ts, sin), 50e-9,
                                                                           fixed_Dt_MESS, )
    df_o33, df_o23,  = [df_outs[i] for i in range(compressor.nw_charging.nports)]
    index_of_outs = list(range(1, compressor.nw_charging.nports))
    axs[i].plot(df_o23[0] / 1e-9,
             # 1- df_o33[key_complex].abs() ** 2 ,
                0.5*(50e6/2830)**2*  G_cav_deembedded **0 *  numpy.array([df_outs[i][key_complex].abs() ** 2 for i in index_of_outs]).sum(axis=0),
             label="traditional"
             )

axs[i].legend()
G_cav_deembedded_traditional = G_cav_deembedded




# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\for_paper\SES_trivial8.cst"
cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.cst"
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
gamma_3_at_f0 = scipy.interpolate.interp1d(gamma_3[:,0].real,gamma_3[:,1])(9.3)
logger.info(gamma_3_at_f0)
v_g = C.c ** 2 / (2 * numpy.pi * f_ref / interp1d(gamma_3[:, 0], gamma_3[:, 1], )(f_ref / 1e9).imag)
# gamma_3[:,1] = gamma_3[:,1] .imag*1j# Ignoring attenuation

S33_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S3,3", run_id)
S23_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S2,3", run_id)
# S43_discharging = get_S_parameter_from_CST_proj(proj_discharging, "S4,3", run_id)
# S23_discharging [:,1] =S23_discharging [:,1] + S43_discharging[:,1]
# S23_discharging[:, 1] = 1 - S33_discharging[:, 1]

S33_charging = get_S_parameter_from_CST_proj(proj_charging, "S3,3", run_id)
S23_charging = get_S_parameter_from_CST_proj(proj_charging, "S2,3", run_id)
# S43_charging = get_S_parameter_from_CST_proj(proj_charging, "S4,3", run_id)
# S23_charging[:, 1] += S43_charging[:, 1]
# S23_charging [:,1] = 1-S33_charging[:,1]
dt = 1 / f_ref / 15.  # 1/f_target / 10
ts = numpy.arange(-20e-9, 0, dt)
# ts = ts[:(len(ts)//2)*2]
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006000000000000015e9)
resampled_f = numpy.arange(9e9, 9.6e9,0.0006e9)
# resampled_f = numpy.arange(8e9, 10.5e9, 0.0006e9)
# resampled_f = numpy.arange(8.5e9, 10e9, 0.0006e9)
# resampled_f = numpy.arange(0e9, 12e9,0.1e9)
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006100000000000015e9)
# compressor.nw_charging = compressor.nw_charging.interpolate(resampled_f)#
# compressor.nw_discharging = compressor.nw_discharging.interpolate(resampled_f)#[band]
compressor = Compressor(
    build_network_from_CST_S_data_1inNout([S33_discharging, S23_discharging, #S43_discharging
                                           ]).interpolate(
        resampled_f)
    #   .extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))
    ,
    build_network_from_CST_S_data_1inNout([S33_charging, S23_charging,# S43_charging
                                           ]).interpolate(
        resampled_f)
    #  .extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))
    ,
)

ts_for_caching_impulse_response = numpy.arange(-400e-9, 0, dt)

L_waveguide=  -1*((300 - 114) ) * 1e-3
compressor_deembedded = compressor.embed_with_waveguide(gamma_3,L_waveguide)
# compressor_deembedded = compressor.embed_with_waveguide(gamma_3, -((300 - 40 - 20 * 0)) * 1e-3)
G_cav_deembedded = (1 - (numpy.abs(compressor_deembedded.nw_charging.interpolate([f_ref]).s[0, 0, 0]) ** 2+loss_of_ignored_high_order_mode)) ** -1
# G_cav_deembedded = (1 - numpy.abs(compressor.nw_charging.interpolate([f_ref]).s[0, 0, 0]) ** 2 * numpy.exp(-4 *  gamma_3_at_f0.real * L_waveguide)) ** -1
logger.info(G_cav_deembedded)
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
# df_i1_charging, df_i1_discharging, df_outs = compressor.run((ts, sin), 10e-9,
#                                                             compressor.correct_Dt_MESS(
#                                                                 f_ref, 15e-9))
# df_o33, df_o23,  = [df_outs[i] for i in range(compressor.nw_charging.nports)]
# plt.figure()
# plt.plot(df_o23[0], df_o23[1])

i=1

for Dt_MESS in numpy.array([0]

                           ):
    fixed_Dt_MESS = compressor_deembedded.correct_Dt_MESS(
        f_ref, Dt_MESS)
    df_i1_charging, df_i1_discharging, df_outs = compressor_deembedded.run((ts, sin), 50e-9,
                                                                           fixed_Dt_MESS, )
    df_o33, df_o23,  = [df_outs[i] for i in range(compressor.nw_charging.nports)]
    index_of_outs = list(range(1, compressor.nw_charging.nports))
    axs[i].plot(df_o23[0] / 1e-9,
             # 1- df_o33[key_complex].abs() ** 2 ,
            0.5*  (50e6/540)**2*  G_cav_deembedded **0*  numpy.array([df_outs[i][key_complex].abs() ** 2 for i in index_of_outs]).sum(axis=0),
             label="with common switch cavity"
             )

axs[i].legend()






# cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\for_paper\SES_MPC_2_way.lower_Efield_when_ES.cst"
cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_2_way.lower_Efield_when_ES.cst"
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
gamma_3_at_f0 = scipy.interpolate.interp1d(gamma_3[:,0].real,gamma_3[:,1])(9.3)
logger.info(gamma_3_at_f0)


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
ts = numpy.arange(-20e-9, 0, dt)
# ts = ts[:(len(ts)//2)*2]
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006000000000000015e9)
# resampled_f = numpy.arange(9e9, 9.6e9,0.0006e9)
resampled_f = numpy.arange(8e9, 10.5e9, 0.0006e9)
# resampled_f = numpy.arange(8.5e9, 10e9, 0.0006e9)
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

ts_for_caching_impulse_response = numpy.arange(-400e-9, 0, dt)

L_waveguide=  -1*((300 - 114) ) * 1e-3
compressor_deembedded = compressor.embed_with_waveguide(gamma_3,L_waveguide)
# compressor_deembedded = compressor.embed_with_waveguide(gamma_3, -((300 - 40 - 20 * 0)) * 1e-3)
G_cav_deembedded = (1 - (numpy.abs(compressor_deembedded.nw_charging.interpolate([f_ref]).s[0, 0, 0]) ** 2+loss_of_ignored_high_order_mode)) ** -1
# G_cav_deembedded = (1 - numpy.abs(compressor.nw_charging.interpolate([f_ref]).s[0, 0, 0]) ** 2 * numpy.exp(-4 *  gamma_3_at_f0.real * L_waveguide)) ** -1
logger.info(G_cav_deembedded)
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
# df_i1_charging, df_i1_discharging, df_outs = compressor.run((ts, sin), 10e-9,
#                                                             compressor.correct_Dt_MESS(
#                                                                 f_ref, 15e-9))
# df_o33, df_o23, df_o43 = [df_outs[i] for i in range(compressor.nw_charging.nports)]
# plt.figure()
# plt.plot(df_o23[0], df_o23[1])

i= 2

for Dt_MESS in numpy.array([0]

                           ):
    fixed_Dt_MESS = compressor_deembedded.correct_Dt_MESS(
        f_ref, Dt_MESS)
    df_i1_charging, df_i1_discharging, df_outs = compressor_deembedded.run((ts, sin), 50e-9,
                                                                           fixed_Dt_MESS, )
    df_o33, df_o23, df_o43 = [df_outs[i] for i in range(compressor.nw_charging.nports)]
    index_of_outs = list(range(1, compressor.nw_charging.nports))
    axs[i].plot(df_o23[0] / 1e-9,
             # 1- df_o33[key_complex].abs() ** 2 ,
                0.5*(50e6/540)**2*  G_cav_deembedded **0*    numpy.array([df_outs[i][key_complex].abs() ** 2 for i in index_of_outs]).sum(axis=0),
             label="the proposed"
             )

fig:plt.Figure
fig.supxlabel("time (ns)")

fig.supylabel("output power gain")
plt.xlim(-1,13)
# plt.ylim(-10,300)
plt.axhline(G_cav_deembedded_traditional, ls= ':',label = r'$G_{cav}(L_{ESWG}=114~mm)$, traditional')
plt.legend()
plt.savefig("out_power_gain.svg")
