# -*- coding: utf-8 -*-
# @Time    : 2025/2/5 15:13
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : why_coaxial.py
# @Software: PyCharm

from _logging import logger
from theory.emission.space_charge_limit import *
from matplotlib.colors import TABLEAU_COLORS

color_table = list(TABLEAU_COLORS.keys())

logger.info(
    HPMJamesBenford2009.I_SCL_unit_in_A_cylindrial_beam_inside_hollow_waveguide(common.Ek_to_gamma(50e3), 10e-3, 2e-3))
logger.info(
    HPMJamesBenford2009.I_SCL_unit_in_A_ring_shaped_beam_inside_hollow_waveguide(common.Ek_to_gamma(50e3), 100e-3,
                                                                                 90e-3))
logger.info(CoaxialAndHollow.I_space_charge_limit_hollow(50e3, 100e-3, 90e-3))

Ek = 50e3
dr_beam = 4e-3
rbeam_out = numpy.linspace(dr_beam + 1e-3, 45e-3, 100)
rbeam_in = rbeam_out - dr_beam
rb = (rbeam_out + rbeam_in) / 2
rout = rbeam_out + 6e-3
rin = rbeam_in - 6e-3

gamma_0 = common.Ek_to_gamma(Ek)
mm = 1e-3
kA = 1e3

plt.figure(figsize=(4, 3), constrained_layout=True)
plt.plot(rbeam_out / mm, ZhangWei2022Doctor.I_SCL_solid_beam(gamma_0, rout, rbeam_out) / kA, label="solid")
plt.plot(rbeam_out / mm,
         ZhangWei2022Doctor.I_SCL_annular_beam_inside_hollow_drift(gamma_0, rout, rbeam_out, rbeam_in) / kA,
         label="ring, hollow")
plt.plot(rbeam_out / mm,
         ZhangWei2022Doctor.I_SCL_annular_beam_inside_coaxial_drift(gamma_0, rout, rbeam_out, rbeam_in, rin) / kA,
         label="coaxial")

# 比较实心束和环形束的驱动束流的功率密度
lambda_0 = C.c / 9.46e9
k1 = 2.405 / numpy.pi
I_SCL_solid_combine = ZhangWei2022Doctor.I_SCL_solid_beam(gamma_0, k1 * lambda_0 / 2, dr_beam) * (
        rb + lambda_0 / 4) ** 2 / (k1 * lambda_0 / 2) ** 2

plt.plot(rbeam_out / mm, I_SCL_solid_combine / kA, label="combine of solid beam")

plt.legend()
plt.xlabel("$r_b$ (mm)")
plt.ylabel("$I_{SCL}$ (kA)")

__R_out_FOR_PLOT, __DR_BEAM_FOR_PLOT = numpy.meshgrid(numpy.linspace(20e-3, 60e-3, 50),
                                                      numpy.linspace(0.1e-3, 6e-3, 50))
# __RBEAM = __R_out_FOR_PLOT - __DR_BEAM_FOR_PLOT / 2
Delta_I = (ZhangWei2022Doctor.I_SCL_annular_beam_inside_coaxial_drift(
    gamma_0, __R_out_FOR_PLOT,
    __R_out_FOR_PLOT - lambda_0 / 4 + __DR_BEAM_FOR_PLOT / 2,
    __R_out_FOR_PLOT - lambda_0 / 4 - __DR_BEAM_FOR_PLOT / 2,
    __R_out_FOR_PLOT - lambda_0 / 2
) -
           ZhangWei2022Doctor.I_SCL_solid_beam(gamma_0, k1 * lambda_0 / 2, __DR_BEAM_FOR_PLOT) * (
                   __R_out_FOR_PLOT
           ) ** 2 / (k1 * lambda_0 / 2) ** 2)
plt.figure(figsize=(4, 3), constrained_layout=True)
cf = plt.contourf(__R_out_FOR_PLOT / mm, __DR_BEAM_FOR_PLOT / mm, Delta_I / kA,
                  levels=20, cmap='jet'
                  )
c = plt.contour(__R_out_FOR_PLOT / mm, __DR_BEAM_FOR_PLOT / mm, Delta_I / kA, levels=[0]
                )
plt.colorbar(cf, label=r"$\Delta I_{SCL} \ (kA)$")
plt.xlabel("bounding radius $R$ (mm)")
plt.ylabel("beam thickness $\Delta$ (mm)")
# plt.xlim(12, None)
plt.clabel(c, inline=True, )
__paramter_used_in_manuscript = (45 +8, 4)
# plt.scatter(*__paramter_used_in_manuscript)
# plt.plot((__paramter_used_in_manuscript[0], __paramter_used_in_manuscript[0]), (0, __paramter_used_in_manuscript[1]),
#          ls='--', c='k')
# plt.plot((0, __paramter_used_in_manuscript[0]), (__paramter_used_in_manuscript[1], __paramter_used_in_manuscript[1]),
#          ls='--', c='k')
