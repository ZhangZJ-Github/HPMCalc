# -*- coding: utf-8 -*-
# @Time    : 2024/6/29 15:12
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : RF_compressor4.py
# @Software: PyCharm
import cst.results,scipy

import numpy, matplotlib,pandas,scipy
import cst.results
matplotlib.use('tkagg')
import matplotlib .pyplot as plt
from scipy.optimize import curve_fit
plt.ion()
# omega = numpy.linspace(8.0e9,10e9,100)
# Q0 = 20
# omega0 = 9.e9
# Chi = Q0*(omega/omega0-omega0/omega0)
# plt.plot(omega, numpy.abs(1/ (1+Chi**2)**0.5))
f_target = 9.3e9


key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
def to_df(o23):
    df_o23 = pandas.DataFrame(o23)
    df_o23[key_period_for_calculate_avg] = df_o23[0] * 1e-9 // (5 / f_target)
    df_o23[key_square] = df_o23[1] ** 2
    df_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
    df_o23['interpolated_periodic_avg_square'] = numpy.interp(df_o23[0], df_time_avg[0], df_time_avg['square'],)
    df_o23[key_complex] = scipy.signal.hilbert(df_o23[1])
    return df_o23

path_schematic = r"E:\CSTprojects\GeneratorAccelerator\HP_SES_out_module_eq_circuit.cst"

proj: cst.results.ProjectFile = cst.results.ProjectFile(path_schematic,
                                                        allow_interactive=True)
# o23 = numpy.array(proj_discharging.get_3d().get_result_item(r'1D Results\Port signals\o2,3[1.0,0.0,signal1],[f_target]').get_data())
# o33 = numpy.array(proj_discharging.get_3d().get_result_item(r'1D Results\Port signals\o3,3[1.0,0.0,signal1],[f_target]').get_data())
# i3 = numpy.array(proj_discharging.get_3d().get_result_item(r'1D Results\Port signals\i3[1.0,0.0,signal1],[9.3]').get_data())
proj_3D :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch01.1.cst",
                                                        allow_interactive=True)
S11 =numpy.array(proj.get_schematic().get_result_item('Tasks\\SPara1\\S-Parameters\\S1,1').get_data())
S21 =numpy.array(proj.get_schematic().get_result_item('Tasks\\SPara1\\S-Parameters\\S1,1').get_data())


S11_3D  =numpy.array(proj_3D.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3',).get_data())
S21_3D  =numpy.array(proj_3D.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3',).get_data())

from cst.interface import DesignEnvironment
de = DesignEnvironment(mode=DesignEnvironment.StartMode.Existing,)
proj_schematic_interface = de.get_open_project(path_schematic)

