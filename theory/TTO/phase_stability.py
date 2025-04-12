# -*- coding: utf-8 -*-
# @Time    : 2025/3/21 13:47
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : data_processing.py
# @Software: PyCharm

import matplotlib

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



from simulation.task_manager.CST_helper import get_interpolator

if __name__ == '__main__':
    proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
        r"E:\CSTprojects\Genac2\CoaxialRFsource.buncher.PIC.hot.cst",
        # r"E:\CSTprojects\Genac2\CoaxialRFsource.buncher.PIC.cold.cst",

        allow_interactive=True)
    run_id= 0
    V_data = numpy.array(
        proj_3D.get_3d().get_result_item('1D Results\\Discrete Ports\\Voltages\\Signals\\Port 101 [pic]',
                                         # 'Tables\\1D Results\\e-field (f=9.3) (pic)_Z (Z)',
                                         run_id).get_data())
    from scipy.signal import hilbert
    from  scipy.fft import fft,fftfreq
    t = V_data[:,0]
    freqs_GHz = fftfreq(len(t),t[1]-t[0])
    fft_data = fft(V_data[:, 1],)
    plt.figure()
    # plt.plot(fft_data[:,0],numpy.abs(fft_data[:,1]))
    plt.plot(freqs_GHz,numpy.abs(fft_data))


    f_GHz =freqs_GHz [numpy.argmax(numpy.abs(fft_data))]
    V_data_as_complex = hilbert(V_data[:,1])

    plt.figure()

    plt.plot(t,numpy.unwrap(numpy.rad2deg(numpy.angle(V_data_as_complex))))
    plt.plot(t, 2*numpy.pi* f_GHz *t)
    plt.plot(t,numpy.unwrap(numpy.rad2deg(numpy.angle(V_data_as_complex)))- 2*numpy.pi* f_GHz *t)


    plt.figure()
    plt.plot(t,numpy.abs(V_data_as_complex ) * numpy.cos(2 * numpy.pi *f_GHz * t + numpy.angle(V_data_as_complex)))
    plt.plot(V_data[:,0],V_data[:,1])


