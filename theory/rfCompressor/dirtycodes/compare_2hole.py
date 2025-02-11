# -*- coding: utf-8 -*-
# @Time    : 2024/11/12 23:36
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : compare_single_hole.py
# @Software: PyCharm
import os.path

import matplotlib.pyplot as plt

from  theory.rfCompressor.time_dependent_output import *
import scipy.constants as C

# cst_proj_paths = [r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial%d.cst"%i for i in [6,10,7,8,9]]  + [r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial.wider_switch_cav.cst"]
cst_proj_paths_dict = {
#     r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.cst":"1hole-1",
# # r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial7.cst":"1hole-1.1",
#     r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.1.cst":"1hole-2",
    r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.wider_switchCav.cst": "1hole-3",
    r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_2_way.lower_Efield_when_ES.normal_GDT.cst": "2hole-1",
    r"E:\CSTprojects\rfCompressor\cascadedHT\SES_MPC_2_way.lower_Efield_when_ES.cst": "2hole-2",
}
cst_proj_paths = list(cst_proj_paths_dict.keys())

dict_projs_discharging: cst.results.ProjectFile = {cst_proj_path:cst.results.ProjectFile(cst_proj_path,
                                                                                         allow_interactive=True) for cst_proj_path in cst_proj_paths}
dict_projs_charging: cst.results.ProjectFile ={ proj_discharging: cst.results.ProjectFile(
    proj_discharging[:
                              # -len("paramsweep.cst")
                              -len("cst")
    ]
    + "ES.cst",
    allow_interactive=True) for proj_discharging in dict_projs_discharging }
run_id = 0
run_id_charging = 0

dict_S33_discharging = {proj_discharging:
    get_S_parameter_from_CST_proj(dict_projs_discharging[proj_discharging], "S3,3", run_id) for proj_discharging in
                        dict_projs_discharging}
dict_S23_discharging ={proj_discharging:get_S_parameter_from_CST_proj(dict_projs_discharging[proj_discharging], "S2,3", run_id) for proj_discharging in dict_projs_discharging}
dict_S43_discharging ={}
for cst_proj_path in cst_proj_paths:
    if cst_proj_paths_dict[cst_proj_path ] .startswith( '2hole'):
        dict_S43_discharging [cst_proj_path ] = get_S_parameter_from_CST_proj(dict_projs_discharging[cst_proj_path], "S4,3", run_id)

dict_S33_charging = {proj_discharging:
    get_S_parameter_from_CST_proj(dict_projs_discharging[proj_discharging], "S3,3", run_id) for proj_discharging in
                        dict_projs_charging}
dict_S23_charging ={proj_discharging:get_S_parameter_from_CST_proj(dict_projs_discharging[proj_discharging], "S2,3", run_id) for proj_discharging in dict_projs_charging}


gamma_3 = numpy.array(
    dict_projs_charging[cst_proj_paths[0]].get_3d().get_result_item('1D Results\\Port Information\\Gamma\\3(1)', run_id_charging).get_data())
v_p = (2 * numpy.pi * f_ref / interp1d(gamma_3[:, 0], gamma_3[:, 1], )(f_ref / 1e9).imag)
v_g = C.c ** 2 /v_p
lambda_g =v_p /f_ref
def get_zeta_star (zeta):
    return zeta if zeta<= 0.5 else 1-zeta

plt.figure(figsize=(5,4),constrained_layout = True)

for cst_proj_path in cst_proj_paths:
    kwargs = {}
    # dz1 = projs_discharging[i].get_3d().get_parameter_combination(run_id)['dz1']
    dz11 = dict_projs_discharging[cst_proj_path].get_3d().get_parameter_combination(run_id)['dz11']
    zeta_star = get_zeta_star((dz11*1e-3 / (lambda_g/2))%1)

    model_alias =  cst_proj_paths_dict[cst_proj_path]
    if  model_alias== "1hole-3" :
    # if numpy.abs(dz1-24) < 0.1:
    #     kwargs =  {"lw" : 5}
        plt.plot(dict_S23_discharging[cst_proj_path][:, 0].real,numpy.abs(dict_S23_discharging[cst_proj_path][:, 1]) ** 1,  #label =os.path.split( projs_discharging[i].filename)[1]
             label = r"%s: $\sqrt{ \sum_{i\ne1}|{S_{i, 1}|^2}}$"%(model_alias),
             # **{'ls':  ':'} ,
                 **kwargs,
                 # lw= 3

             )

    if model_alias ==  "2hole-1"  :
        plt.plot(dict_S23_discharging[cst_proj_path][:, 0].real,
                 (numpy.abs(dict_S23_discharging[cst_proj_path][:, 1]) ** 2 + numpy.abs(
                     dict_S43_discharging[cst_proj_path][:, 1]) ** 2) ** 0.5,
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"%s: $\sqrt{ \sum_{i\ne1}|{S_{i, 1}|^2}}$" % (model_alias,),
                 # lw=3,#ls = ':',
                 **kwargs
                 )

    if model_alias.startswith("2hole-2"):
        plt.plot(dict_S23_discharging[cst_proj_path][:, 0].real,
        (numpy.abs(dict_S23_discharging[cst_proj_path][:, 1]) ** 2 + numpy.abs(dict_S43_discharging[cst_proj_path][:, 1]) ** 2)**0.5,
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"%s: $\sqrt{ \sum_{i\ne1}|{S_{i, 1}|^2}}$" % (model_alias, ),
                 lw=  3,
                 **kwargs
                 )

        plt.plot(dict_S23_discharging[cst_proj_path][:, 0].real,
        (numpy.abs(dict_S23_discharging[cst_proj_path][:, 1]) ** 2  )**0.5,
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"%s: $|S_{2,1}|$" % (model_alias, ),
                 ls=':',
                 **kwargs
                 )
        plt.plot(dict_S23_discharging[cst_proj_path][:, 0].real,
        (  numpy.abs(dict_S43_discharging[cst_proj_path][:, 1]) ** 2)**0.5,
                 # label =os.path.split( projs_discharging[i].filename)[1]
                 label=r"%s: $|S_{3,1}|$" % (model_alias, ),
                 ls = ':',
                 **kwargs
                 )


plt.ylim([0, 1.05])
# plt.xlim([8.5, 10.])


plt.legend(loc = "upper right")
plt.xlabel("frequency (GHz)")
plt.ylabel("scattering parameter")
plt.savefig("S21_1holes.svg")