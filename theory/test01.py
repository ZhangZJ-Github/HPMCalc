# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 22:15
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : test01.py
# @Software: PyCharm
import numpy
import numpy as np
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

from scipy.signal import hilbert, chirp
plt.ion()
duration = 1.0
fs = 400.0 # 采样率
samples = int(fs*duration)
t = np.arange(samples) / fs

signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)
fig, (ax0, ax1,ax2) = plt.subplots(nrows=3)
ax0.plot(t, signal, label='signal')
ax0.plot(t, analytic_signal.real,label = 'hilbert')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency)

ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)

ax2:plt.Axes
ax2.plot(t, numpy.angle(analytic_signal))
ax2.plot(t, numpy.unwrap(numpy.angle(analytic_signal)))


fig.tight_layout()