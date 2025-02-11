# -*- coding: utf-8 -*-
# @Time    : 2025/2/5 15:13
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : why_coaxial.py
# @Software: PyCharm

from _logging import logger
from theory.emission.space_charge_limit import *

logger.info(
    HPMJamesBenford2009.I_SCL_unit_in_A_cylindrial_beam_inside_hollow_waveguide(common.Ek_to_gamma(50e3), 10e-3, 2e-3))
logger.info(
    HPMJamesBenford2009.I_SCL_unit_in_A_ring_shaped_beam_inside_hollow_waveguide(common.Ek_to_gamma(50e3), 100e-3,
                                                                                 90e-3))
logger.info(CoaxialAndHollow.I_space_charge_limit_hollow(50e3, 100e-3, 90e-3))

Ek = 50e3
dr_beam = 4e-3
rbeam_out = numpy.linspace(dr_beam +1e-3, 45e-3, 100)
rbeam_in = rbeam_out - dr_beam
rout = rbeam_out + 3e-3
rin = rbeam_in - 3e-3

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
         label="ring, coaxial")
plt.legend()
plt.xlabel("$r_b$ (mm)")
plt.ylabel("$I_{SCL}$ (kA)")
