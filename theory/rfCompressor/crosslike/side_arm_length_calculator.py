# -*- coding: utf-8 -*-
# @Time    : 2025/1/11 21:16
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : side_arm_length_calculator.py
# @Software: PyCharm
import numpy
import cst.results,scipy
import skrf
import numpy, matplotlib,pandas,scipy
import cst.results
from scipy.interpolate import interp1d
matplotlib.use('tkagg')
import matplotlib .pyplot as plt
from scipy.optimize import curve_fit
plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_target= 9.3e9
f_target_GHz =f_target / 1e9
def get_beta_r4_to_eliminate_o3 ( S31,S41,S34,S44
        ):
    """
    问题描述：
    对于包含1, 3, 4三端口的网络S，
    1端口输入，
    3端口接匹配负载（输出），
    4端口（支臂）接长度为r4的波导，且终端短路，
    求beta * r4的值，使3端口输出为0
    其中 beta = 2 pi / lambda_g 为支臂波导的纵向波数

    :param S_II_34:
    :return:
    """
    Gamma = S31 / (S31*S44 - S34* S41)
    beta_r4 = numpy.arctan( 1+Gamma/ (1j*(1-Gamma)))
    return beta_r4
def get_o3(S31,S41, S34,S44,Gamma,i1= 1+0j):
    return (S31+ Gamma * S34 * S41 / (1-S44 * Gamma)) * i1

def get_interpolator (E_data):
    return interp1d(E_data[:,0].real,E_data[:,1])
if __name__ == '__main__':
    proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
        r"E:\CSTprojects\rfCompressor\cascadedHT\why_2hole\HplaneCrossLike.cst",
        allow_interactive=True)


    # proj_discharging :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch02.cst",
    #                                                         allow_interactive=True)
    # proj_3D_charging :cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch02.1.cst",
    #                                                         allow_interactive=True)
    S31 = numpy.array(proj_3D.get_schematic().get_result_item( 'Tasks\\SPara1\\S-Parameters\\S3,1', ).get_data())
    S41 = numpy.array(proj_3D.get_schematic().get_result_item( 'Tasks\\SPara1\\S-Parameters\\S4,1', ).get_data())
    S34 = numpy.array(proj_3D.get_schematic().get_result_item( 'Tasks\\SPara1\\S-Parameters\\S3,4', ).get_data())
    S44 = numpy.array(proj_3D.get_schematic().get_result_item( 'Tasks\\SPara1\\S-Parameters\\S4,4', ).get_data())
    S31_interpolator = get_interpolator(S31)
    S41_interpolator = get_interpolator(S41)
    S34_interpolator = get_interpolator(S34)
    S44_interpolator = get_interpolator(S44)

    # plt.plot(S34[:, 0], S34[:,1] * S41[:,1])
    # phi = numpy.linspace(0 , 4*numpy.pi ,720)
    beta_r4 = numpy.linspace(0 , 2*numpy.pi ,720)
    z4 = 1j*numpy.tan(beta_r4)
    Gamma = (z4-1)/(z4+1)
    plt.figure()
    plt.plot(beta_r4/(2 * numpy.pi) ,        numpy.abs(get_o3(S31_interpolator(f_target_GHz),S41_interpolator(f_target_GHz),S34_interpolator(f_target_GHz),S44_interpolator(f_target_GHz),Gamma)))
