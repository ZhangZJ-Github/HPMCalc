# -*- coding: utf-8 -*-
# @Time    : 2025/3/19 15:46
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : estimate_Q_cold_PIC.py
# @Software: PyCharm


import matplotlib
import scipy.interpolate

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()
import common
import numpy
import scipy.constants as C

ln = numpy.log

import cst.results
import cst.results
import matplotlib

matplotlib.use('tkagg')

from _logging import logger

import numpy
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_target = 9.3e9
f_target_GHz = f_target / 1e9


def get_interpolator(E_data):
    return interp1d(E_data[:, 0].real, E_data[:, 1])


if __name__ == '__main__':
    proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
        r"E:\CSTprojects\Genac2\CoaxialRFsource.buncher.PIC.cold.test_Q.cst",
        # r"E:\CSTprojects\Genac2\CoaxialRFsource.buncher.PIC.cold.2.cst",

        allow_interactive=True)
    run_id = 0

    out_signal = numpy.array(
        proj_3D.get_3d().get_result_item('1D Results\\Port signals\\o1(1),pic',
                                         # 'Tables\\1D Results\\e-field (f=9.341) (pic)_Z (Z)',
                                         run_id).get_data())
    from scipy.signal import hilbert
    out_signal_amp = numpy.abs(hilbert(out_signal[:, 1]))
    ts= out_signal[:, 0]
    out_signal_amp_interp = scipy.interpolate.interp1d(ts,out_signal_amp)
    plt.figure()
    plt.plot(ts, out_signal_amp,label = "Original data")


    def decay (t, E0,t0, tau):
        return E0 *  numpy.exp(-(t- t0 )/tau)
    from scipy.optimize import curve_fit
    trusted_indexes = (ts>45)& (ts<95)
    (E0, t0, tau),_=    curve_fit(decay, ts[trusted_indexes], out_signal_amp[trusted_indexes],p0 = [out_signal.max(),45,40])
    time_unit = 1e-9
    f0 = 9.3e9
    Q = 2* numpy.pi *f0 * tau * time_unit / 2
    logger.info(Q)
    plt.plot(ts[trusted_indexes], decay(ts[trusted_indexes],E0, t0, tau ),label = "Fitted, Q = %.2f"%Q)
    plt.legend()
    plt.xlabel("time (ns)")
    plt.ylabel("amp. (a.u.)")




