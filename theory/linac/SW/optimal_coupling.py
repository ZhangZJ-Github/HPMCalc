# -*- coding: utf-8 -*-
# @Time    : 2025/7/23 13:59
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : optimal_coupling.py
# @Software: PyCharm
import numpy

from _logging import logger
import common
import scipy.constants as C
def get_optimal_coupling_coefficient(I, ZL,P_input):
    """
    计算驻波加速管的最佳外部耦合度\beta_{opt}
    Ref:

    Gao, Jian, Hao Zha, Jia-Ru Shi, 等. 《Design, Fabrication, and Testing of an X-Band 9-MeV Standing-Wave Electron Linear Accelerator》. Nuclear Science and Techniques 34, 期 7 (2023年): 110. https://doi.org/10.1007/s41365-023-01254-8.

    Eqn. (5)

    :return:
    """
    beta_opt = (I/ 2 * (ZL /P_input) ** .5 + (1+I**2 * ZL / (4*P_input))**0.5)**2
    return beta_opt

def estimate_effective_accelerating_voltage_considering_beam_loading(beta_0 , P_input, LZTT,I_b,
                                                                     ):
    """
    适用条件：最佳耦合状态，且电子束加速相位 = 0 （波峰加速），加速管冷腔失谐 = 0
    其中LZTT = 总长L * 有效分路阻抗ZTT
    有效分路阻抗采用加速器的定义，即ZTT = |V_acc|^2 /P_wall
    Refs:

    Eqn. (1) in

    Lee, Yong‐Seok, Sanghoon Kim, Geun‐Ju Kim, 等. 《Medical X‐band Linear Accelerator for High‐precision Radiotherapy》. Medical Physics 48, 期 9 (2021年): 5327～42. https://doi.org/10.1002/mp.15077.


    Eqn. （4）-（21) in

    Liu, Focheng, Jiaru Shi, Hao Zha, Qiang Gao, Huaibi Chen和Jiaqi Qiu. 《Transient Study of Beam Instability Due to Beam Loading in Standing-Wave Low-Energy Electron Linear Accelerators》. Physical Review Accelerators and Beams 28, 期 5 (2025年): 050101. https://doi.org/10.1103/PhysRevAccelBeams.28.050101.


    :return:
    """
    return (2* ((beta_0* P_input*LZTT)**0.5/(1+beta_0) ))-LZTT*I_b/(1+beta_0)



# Gao, Jian, Hao Zha, Jia-Ru Shi, 等. 《Design, Fabrication, and Testing of an X-Band 9-MeV Standing-Wave Electron Linear Accelerator》. Nuclear Science and Techniques 34, 期 7 (2023年): 110. https://doi.org/10.1007/s41365-023-01254-8.
logger.info(get_optimal_coupling_coefficient(0.1 * 0.32,165e6 * 0.59,2.4e6,))
# 牟启航, 李杰成, 柳淘, 等. 《X波段2 MeV小焦点加速器研制及应用》. 原子核物理评论 41, 期 1 (2024年): 418～25.
logger.info(get_optimal_coupling_coefficient(0.13,71e6 * 95e-3,(0.6 + 2 * 0.13) *1e6,))


L = 303e-3#315e-3
Zeff = 125e6#119e6
Ibeam = 0.1
Pin = 5.67e6
# Pwall = 4.5e6#5.8e6
beta_opt = get_optimal_coupling_coefficient(Ibeam,Zeff * L,Pin,)
V_acc_considering_beam_loading = estimate_effective_accelerating_voltage_considering_beam_loading(beta_opt,Pin,Zeff * L,Ibeam)
logger.info("beta_opt = %.8f"%beta_opt)
logger.info("考虑束流负载，有效加速电压 = %.8f MV"%(V_acc_considering_beam_loading/1e6))
Pwall = V_acc_considering_beam_loading**2 / (Zeff*L)
Pext = V_acc_considering_beam_loading**2 / (Zeff*L/(beta_opt))
Pbeam = V_acc_considering_beam_loading * Ibeam
logger.info("Pwall = %.8f MW, Pext = %.8f MW, Pbeam = %.8f"%(
    *(numpy.array([Pwall,Pext,Pbeam])/1e6),
))
logger.info("1+P_b/P_wall = %.8f"%(Pin / Pwall))

# 平均加速梯度
Ea = V_acc_considering_beam_loading/L
logger.info("考虑束流负载，平均加速梯度 = %.8f MV/m"%(Ea/1e6))


estimated_BDR = 1e-6 # per pulse pur meter
N_pulse = 1e4
logger.info("估计的击穿概率 = %.8e"%(
   1- (1 -  estimated_BDR* L )**N_pulse))



