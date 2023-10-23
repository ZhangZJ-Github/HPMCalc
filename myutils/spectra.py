# -*- coding: utf-8 -*-
# @Time    : 2023/10/18 23:43
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : spectra.py
# @Software: PyCharm

# 分析1D频谱
from myutils.cst_exported_txt_parser import *

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy
import pandas
from scipy import signal
from scipy.fftpack import fft
import scipy.constants as C

plt.ion()


def spectrogram(df):
    # dfs, labels = parse(path)

    sample_rate = 1 / numpy.diff(df, axis=0).mean(axis=0)[0]  # 每秒采样次数
    print("sample_rate = %.2e" % sample_rate)
    x = df.iloc[:, 1]

    # Compute and plot the spectrogram.
    freqs, ts, Sxx = signal.spectrogram(x, sample_rate,  # mode='complex',
                                        scaling='spectrum',  # return_onesided=False
                                        nperseg=75,
                                        # nfft=2000
                                        )
    plt.figure()
    plt.pcolormesh(ts, freqs, Sxx + 1e-12, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.figure()
    plt.pcolormesh(ts, 40e9 / (freqs + 1) / C.c, Sxx, shading='gouraud')
    plt.grid()
    plt.ylabel('phase velocity [c]')
    plt.xlabel('z [m]')

    return freqs, ts, Sxx


def plot_signal(df):
    plt.figure()
    plt.plot(df.iloc[:, 0], df.iloc[:, 1])
    plt.title("Signal")
    plt.xlabel("time")


def plot_amp(freqs, ts, Sxx, index_of_timeslice):
    """
    绘制各频率信号的幅值

    :param freqs:
    :param ts:
    :param Sxx:
    :param index_of_timeslice:
    :return:
    """
    plt.figure()
    amps = ((Sxx[:, index_of_timeslice] * 2) ** .5)
    plt.plot(freqs, amps, label="t = %s s" % ts[index_of_timeslice])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("frequency / Hz")
    plt.ylabel("amplitude")
    # print("Total energy = %.2e\n" % ((amps ** 2).sum() / 2 ) ** .5)
    # print(amps)

    plt.legend()


def plot_fft_result(ts, Vs: numpy.ndarray):
    N = len(ts)
    fs = 1 / numpy.diff(ts).mean()
    res = fft(Vs)[:N // 2]
    freqs = [i * fs / N for i in range(len(res))]
    plt.figure()
    amplitudes = numpy.abs(res) / (N // 2)
    amplitudes[0] /= 2
    plt.plot(freqs, amplitudes)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Amplitude")


def basic_test():
    t = numpy.arange(0, 10, 1e-2)
    data = pandas.DataFrame(numpy.array((t, 50 * numpy.sin(2 * numpy.pi * 12 * t)  # + 3400
                                         )).T)
    freqs1, ts1, Sxx1 = spectrogram(data)
    plot_amp(freqs1, ts1, Sxx1, 2)
    plot_fft_result(data.iloc[:, 0], data.iloc[:, 1].ravel())


if __name__ == '__main__':
    # basic_test()

    # path = r"F:\changeworld\temp\acc\test\data\cone-0.1.8.txt"
    path = r"E:\CSTprojects\GeneratorAccelerator\Ez_along_z.txt"
    dfs, labels = parse(path)
    probe_index = 0
    df = dfs[probe_index]

    # To SI unit
    df.iloc[:, 0] *= 1e-3
    df = df.iloc[300:len(df) - 2700, :]  # 去除前后幅值过高的 否则相速度曲线不明显 难以对比

    label = labels[probe_index]
    for i in range(len(label)):
        print(label[i])
    plot_signal(df)

    freqs, ts, Sxx = spectrogram(df)

    index_of_timeslice = 4
    # plot_amp(freqs, ts, Sxx, index_of_timeslice)

    plot_fft_result(df.iloc[:, 0], df.iloc[:, 1].ravel())

    plt.show()
