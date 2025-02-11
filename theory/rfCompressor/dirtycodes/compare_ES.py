# -*- coding: utf-8 -*-
# @Time    : 2024/11/27 20:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : compare_ES.py
# @Software: PyCharm
import enum
import os.path
import typing

import matplotlib.pyplot as plt

from  theory.rfCompressor.time_dependent_output import *
import scipy.constants as C

import os.path



from  theory.rfCompressor.time_dependent_output import *

label_of_2out_model = "the proposed"

# cst_proj_paths = [r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial%d.cst"%i for i in [6,10,7,8,9]]  + [r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial.wider_switch_cav.cst"]
cst_proj_paths_dict = {r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_traditional.cst":"the traditional",
    r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.cst":"with small switch cavity",
r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_2_way.lower_Efield_when_ES.cst":label_of_2out_model,
}
cst_proj_paths = list(cst_proj_paths_dict.keys())
class Regime(enum.Enum):
    charging=  0
    discharging = 1
dict_projs: cst.results.ProjectFile = {Regime.discharging:{cst_proj_path: cst.results.ProjectFile(cst_proj_path,
                                                                               allow_interactive=True) for cst_proj_path
                                        in cst_proj_paths},
                                       Regime.charging:
{ proj_discharging: cst.results.ProjectFile(
    proj_discharging[:
                              # -len("paramsweep.cst")
                              -len("cst")
    ]
    + "ES.cst",
    allow_interactive=True) for proj_discharging in cst_proj_paths }
                                       }

run_id = 0
run_id_charging = 0

dict_S33= {regime:{proj_discharging:
                get_S_parameter_from_CST_proj(dict_projs[regime][proj_discharging], "S3,3", run_id) for
            proj_discharging in
            cst_proj_paths_dict} for regime in Regime           }
dict_S23 ={regime:
     {proj_discharging:get_S_parameter_from_CST_proj(dict_projs[regime][proj_discharging], "S2,3", run_id) for proj_discharging in cst_proj_paths} for regime in Regime

}

def get_S43(regime:Regime):
    d ={}
    for proj_discharging in cst_proj_paths:
        if cst_proj_paths_dict[proj_discharging].startswith(label_of_2out_model):
            d[proj_discharging] = get_S_parameter_from_CST_proj(dict_projs[regime][proj_discharging], "S4,3", run_id)
    return      d
dict_S43 ={regime : get_S43(regime) for regime in Regime}




gamma_3 = numpy.array(
    dict_projs[Regime.charging][cst_proj_paths[0]].get_3d().get_result_item('1D Results\\Port Information\\Gamma\\3(1)', run_id_charging).get_data())
v_p = (2 * numpy.pi * f_ref / interp1d(gamma_3[:, 0], gamma_3[:, 1], )(f_ref / 1e9).imag)
v_g = C.c ** 2 /v_p
lambda_g =v_p /f_ref
def get_zeta_star (zeta):
    return zeta if zeta<= 0.5 else 1-zeta


# 充电过程
fig, ax =plt.subplots(3,1,figsize=(5,4),constrained_layout = True,sharex = True, sharey= True)

for i,cst_proj_path in enumerate(cst_proj_paths):
    kwargs = {}
    regime = Regime.charging
    # dz1 = projs_discharging[i].get_3d().get_parameter_combination(run_id)['dz1']
    # dz11 = dict_projs_discharging[cst_proj_path].get_3d().get_parameter_combination(run_id)['dz11']
    # zeta_star = get_zeta_star((dz11*1e-3 / (lambda_g/2))%1)

    model_alias =  cst_proj_paths_dict[cst_proj_path]

    if model_alias.startswith(label_of_2out_model):
        ax[i].plot(dict_S23[regime][cst_proj_path][:, 0].real,
20*numpy.log10(        (numpy.abs(dict_S23[regime][cst_proj_path][:, 1]) ** 2 + numpy.abs(dict_S43[regime][cst_proj_path][:, 1]) ** 2)**0.5),
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"$\sqrt{ \sum_{i\ne1}|{S_{i, 1}^c|^2}}$" ,
                 # lw=  3,
                 **kwargs
                 )

        ax[i].plot(dict_S23[regime][cst_proj_path][:, 0].real,
20*numpy.log10(        (numpy.abs(dict_S23[regime][cst_proj_path][:, 1]) ** 2  )**0.5),
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"$|S_{2,1}^c|$" ,
                 ls=':',
                 **kwargs
                 )
        ax[i].plot(dict_S23[regime][cst_proj_path][:, 0].real,
        20*numpy.log10((  numpy.abs(dict_S43[regime][cst_proj_path][:, 1]) ** 2)**0.5),
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"$|S_{3,1}^c|$" ,
                 ls = ':',
                 **kwargs
                 )
    else:
        ax[i].plot(dict_S23[regime][cst_proj_path][:, 0].real,
         20*numpy.log10(    (numpy.abs(dict_S23[regime][cst_proj_path][:, 1]) ** 2  ) ** 0.5),
             # label =os.path.split( projs_discharging[i].filename)[1]
             label=r"$\sqrt{ \sum_{i\ne1}|{S_{i, 1}^c|^2}}$" ,
             # lw=3,#ls = ':',
             **kwargs
             )
    ax[i].legend(#loc = "upper right"
                 )
    ax[i].grid()

# ax[i].ylim([0, 1.05])
ax[i].set_ylim([None, 0])
ax[i].set_xlim([9,9.6])



ax[i].set_xlabel("frequency (GHz)")
fig.supylabel("scattering parameter (dB)")

plt.savefig("power_leakage_charging.svg")





# 放电过程
fig, ax =plt.subplots(3,1,figsize=(5,4),constrained_layout = True,sharex = True, sharey= True)

for i,cst_proj_path in enumerate(cst_proj_paths):
    kwargs = {}
    regime = Regime.discharging
    # dz1 = projs_discharging[i].get_3d().get_parameter_combination(run_id)['dz1']
    # dz11 = dict_projs_discharging[cst_proj_path].get_3d().get_parameter_combination(run_id)['dz11']
    # zeta_star = get_zeta_star((dz11*1e-3 / (lambda_g/2))%1)

    model_alias =  cst_proj_paths_dict[cst_proj_path]

    if model_alias.startswith(label_of_2out_model):
        ax[i].plot(dict_S23[regime][cst_proj_path][:, 0].real,
 (        (numpy.abs(dict_S23[regime][cst_proj_path][:, 1]) ** 2 + numpy.abs(dict_S43[regime][cst_proj_path][:, 1]) ** 2)**0.5),
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"$\sqrt{ \sum_{i\ne1}|{S_{i, 1}^d|^2}}$" ,
                 # lw=  3,
                 **kwargs
                 )

        ax[i].plot(dict_S23[regime][cst_proj_path][:, 0].real,
 (        (numpy.abs(dict_S23[regime][cst_proj_path][:, 1]) ** 2  )**0.5),
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"$|S_{2,1}^d|$" ,
                 ls=':',
                 **kwargs
                 )
        ax[i].plot(dict_S23[regime][cst_proj_path][:, 0].real,
         ((  numpy.abs(dict_S43[regime][cst_proj_path][:, 1]) ** 2)**0.5),
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"$|S_{3,1}^d|$" ,
                 ls = ':',
                 **kwargs
                 )
    else:
        ax[i].plot(dict_S23[regime][cst_proj_path][:, 0].real,
       (    (numpy.abs(dict_S23[regime][cst_proj_path][:, 1]) ** 2  ) ** 0.5),
             # label =os.path.split( projs_discharging[i].filename)[1]
             label=r"$\sqrt{ \sum_{i\ne1}|{S_{i, 1}^d|^2}}$" ,
             # lw=3,#ls = ':',
             **kwargs
             )
    ax[i].legend(#loc = "upper right"
                 )
    ax[i].grid()


# ax[i].ylim([0, 1.05])
ax[i].set_ylim([None, 1.05])
ax[i].set_xlim([8.5,10.])



ax[i].set_xlabel("frequency (GHz)")
fig.supylabel("scattering parameter")

plt.savefig("power_leakage_discharging.svg")



plt.figure(figsize=(5,4),constrained_layout = True)

for i,cst_proj_path in enumerate(cst_proj_paths):
    kwargs = {}
    regime = Regime.charging
    # dz1 = projs_discharging[i].get_3d().get_parameter_combination(run_id)['dz1']
    # dz11 = dict_projs_discharging[cst_proj_path].get_3d().get_parameter_combination(run_id)['dz11']
    # zeta_star = get_zeta_star((dz11*1e-3 / (lambda_g/2))%1)

    model_alias =  cst_proj_paths_dict[cst_proj_path]
    plt.plot(dict_S23[regime][cst_proj_path][:, 0].real,
              1/(1-numpy.abs(
                 dict_S33[regime][cst_proj_path][:, 1]) ** 2),
             # label =os.path.split( projs_discharging[i].filename)[1]
             label=r"%s" % (model_alias,),
             # lw=  3,
             **kwargs
             )


plt.ylim(-5, 130)
plt.xlabel('frequency (GHz)')
plt.ylabel('$G_{cav}$ of a 300-mm-long structure')
plt.xlim(9,9.6)
plt.legend()
plt.savefig("G_cav_300mm.svg")

# 比较上升时间
fig, ax =plt.subplots(3,1,figsize=(5,4),constrained_layout = True,
                      sharex = True, sharey= True)
ax:typing.List[plt.Axes]
for i,cst_proj_path in enumerate(cst_proj_paths):
    kwargs = {}
    regime = Regime.discharging
    # dz1 = projs_discharging[i].get_3d().get_parameter_combination(run_id)['dz1']
    # dz11 = dict_projs_discharging[cst_proj_path].get_3d().get_parameter_combination(run_id)['dz11']
    # zeta_star = get_zeta_star((dz11*1e-3 / (lambda_g/2))%1)

    model_alias =  cst_proj_paths_dict[cst_proj_path]

    if model_alias.startswith(label_of_2out_model):
        ax[i].plot()
    ax[i].legend(#loc = "upper right"
                 )
    ax[i].grid()


# ax[i].ylim([0, 1.05])
ax[i].set_ylim([None, 1.05])
ax[i].set_xlim([8.5,10.])



ax[i].set_xlabel("frequency (GHz)")
fig.supylabel("scattering parameter")

plt.savefig("power_leakage_discharging.svg")
