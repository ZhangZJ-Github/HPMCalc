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
    r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.cst":"1hole-1",
# r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial7.cst":"1hole-1.1",
#     r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.1.cst":"1hole-2",
    r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.wider_switchCav.cst":"1hole-3",
#
#     r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.2.cst":"1hole-4",
#     r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.3.cst":"1hole-5",
#
#     r"E:\CSTprojects\rfCompressor\cascadedHT\2hole\SES_2hole1.cst":"2hole-1out",
r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial6.cst":"1hole-noAP",
    # r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.noFDFD.cst":"1hole-noFDFD",
    r"E:\CSTprojects\rfCompressor\cascadedHT\single_coupling_hole\SES_trivial8.noFDFD2.cst":"1hole-noFDFD",
    r"E:\CSTprojects\rfCompressor\cascadedHT\2hole\SES_2hol4.cst":'2hole-4',
    r"E:\CSTprojects\rfCompressor\cascadedHT\2hole\SES_2hol5.cst":'2hole-5',
    r"E:\CSTprojects\rfCompressor\cascadedHT\2hole\SES_2hol6.cst":'2hole-6',

}

cst_proj_paths = list(cst_proj_paths_dict.keys())

projs_discharging: cst.results.ProjectFile = [cst.results.ProjectFile(cst_proj_path,
                                                                    allow_interactive=True) for cst_proj_path in  cst_proj_paths]
projs_charging: cst.results.ProjectFile =[ cst.results.ProjectFile(
    proj_discharging.filename[:
                              # -len("paramsweep.cst")
                              -len("cst")
    ]
    + "ES.cst",
    allow_interactive=True) for proj_discharging in projs_discharging ]
run_id = 0
run_id_charging = 0

list_S33_discharging =[ get_S_parameter_from_CST_proj(proj_discharging , "S3,3", run_id) for proj_discharging in projs_discharging ]
list_S23_discharging =[ get_S_parameter_from_CST_proj(proj_discharging , "S2,3", run_id) for proj_discharging in projs_discharging ]

list_S33_charging =[ get_S_parameter_from_CST_proj(proj_discharging , "S3,3", run_id) for proj_discharging in projs_charging ]
list_S23_charging =[ get_S_parameter_from_CST_proj(proj_discharging , "S2,3", run_id) for proj_discharging in projs_charging ]
gamma_3 = numpy.array(
    projs_charging[0].get_3d().get_result_item('1D Results\\Port Information\\Gamma\\3(1)', run_id_charging).get_data())
v_p = (2 * numpy.pi * f_ref / interp1d(gamma_3[:, 0], gamma_3[:, 1], )(f_ref / 1e9).imag)
v_g = C.c ** 2 /v_p
lambda_g =v_p /f_ref
def get_zeta_star (zeta):
    return zeta if zeta<= 0.5 else 1-zeta

plt.figure(figsize=(5,4),constrained_layout = True)

for i in range(len(projs_discharging)):
    kwargs = {}
    # dz1 = projs_discharging[i].get_3d().get_parameter_combination(run_id)['dz1']
    dz11 = projs_discharging[i].get_3d().get_parameter_combination(run_id)['dz11']
    zeta_star = get_zeta_star((dz11*1e-3 / (lambda_g/2))%1)
    model_alias =  cst_proj_paths_dict[projs_discharging[i].filename]
    if  model_alias== "1hole-1":
    # if numpy.abs(dz1-24) < 0.1:
    #     kwargs =  {"lw" : 5}
        pass
    plt.plot(list_S23_discharging[i][:,0].real,numpy.abs(list_S23_discharging[i][:,1])**1,#label =os.path.split( projs_discharging[i].filename)[1]
             # label = r"%s ($\zeta^*$ = %.2f)"%(model_alias, zeta_star),
             label = r"%s"%(model_alias,# zeta_star
                            ),
             **kwargs
             )
plt.legend()
plt.xlabel("frequency (GHz)")
plt.ylabel("$|S_{2,1}|$")
plt.savefig("S21_1holes.pdf")