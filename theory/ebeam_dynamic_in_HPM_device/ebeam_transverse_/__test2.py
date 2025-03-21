# -*- coding: utf-8 -*-
# @Time    : 2025/2/13 14:08
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : test2.py
# @Software: PyCharm

import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
import common
import numpy
import scipy.constants as C
from scipy.integrate import solve_ivp

ln = numpy.log

from theory.ebeam_dynamic_in_HPM_device.ebeam_transverse_.ebeam_transverse import \
    AnnularBeamInsideCoaxialDriftWeiYuanZhang, AnnularBeamInsideCoaxialDrift, dY_considering_thickness

L_drift = 250e-3
Ek_beam = 50e3
rb = 45e-3
drbeam = 4e-3
rbin, rbout = rb - drbeam / 2, rb + drbeam / 2  # 电子束内外半径
I0 = 120
q = C.e
rin = 36e-3
rout = 51e-3  # 48e-3
Bz = 0.1

vz = common.Ek_to_beta(Ek_beam) * C.c
gamma = common.Ek_to_gamma(Ek_beam)
beta = vz / C.c
ts = numpy.linspace(0, L_drift / vz, 2000)

__rs_for_plot = numpy.linspace(rin, rout, 300)
ab_wei = AnnularBeamInsideCoaxialDriftWeiYuanZhang(rout, rbout, rbin, rin)
ab_ling = AnnularBeamInsideCoaxialDrift
plt.figure(3)
plt.plot(__rs_for_plot, ab_wei.E_sc_r(__rs_for_plot, I0, vz), label="considering thickness")
plt.plot(__rs_for_plot, ab_ling.E_SC(__rs_for_plot, I0, rout, rb, rin, vz), label="0-thickness approximation")
plt.legend()

sol = solve_ivp(lambda t, r__theta__D_r__r2_D_theta: dY_considering_thickness(
    t,
    *r__theta__D_r__r2_D_theta,
    ab_wei,
    I0, vz,
   q, C.m_e * gamma, beta,
    Bz
), [ts[0], ts[-1]],
                numpy.array([rb, 0, 0, 0]), t_eval=ts,
                # args=(I0, rout, rb, rin, vz,
                #       C.e, C.m_e, beta,
                #       Bz)
                # method="Radau"
                )

r = sol.y[0, :]
theta = sol.y[1, :]
mm = 1e-3
plt.figure(1)
plt.axhline(48e-3 / mm, ls=":")
plt.plot(vz * ts / mm, r / mm, label="B = %.2f T" % Bz)
plt.legend()
# plt.ylim(-1.5 * rb /mm, 1.5* rb /mm)
# plt.gca().set_aspect("equal")
plt.xlabel("z (mm)")

plt.figure(2)
plt.plot(r * numpy.cos(theta) / mm, r * numpy.sin(theta) / mm, label="B = %.2f T" % Bz)
plt.legend()
plt.xlim(-1.5 * rb / mm, 1.5 * rb / mm)
plt.ylim(-1.5 * rb / mm, 1.5 * rb / mm)
plt.gca().set_aspect("equal")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
