# -*- coding: utf-8 -*-
# @Time    : 2023/6/20 16:12
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : find_peaks.py
# @Software: PyCharm

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

import grd_parser
import numpy


class GaussPeakSeeker:
    def __init__(self, N_peaks, data: numpy.ndarray):
        self.N_peaks = N_peaks
        self.fit_res = self.Gauss_fit_N_peak(data, N_peaks)

    @staticmethod
    def Gauss_N_peak(x, As, mus, sigmas):
        # mus =[mu1,mu2,mu3]
        # As =[A1,A2,A3]
        # sigmas = [sigma1,sigma2,sigma3]
        res = 0
        for i in range(len(As)):
            res += As[i] * numpy.exp(- ((x - mus[i]) / sigmas[i]) ** 2 / 2)
        return res

    @staticmethod
    def _Gauss_N_peak_for_curve_fit(x, N_peaks, *args):
        As = args[:N_peaks]
        mus = args[N_peaks:2 * N_peaks]
        sigmas = args[2 * N_peaks:3 * N_peaks]

        return GaussPeakSeeker.Gauss_N_peak(x, As, mus, sigmas)

    @staticmethod
    def Gauss_fit_N_peak(data: numpy.ndarray, N_peaks):
        """
        拟合为N个峰的高斯函数
        :return:
        """
        from scipy.optimize import curve_fit

        return curve_fit(lambda x, *args: GaussPeakSeeker._Gauss_N_peak_for_curve_fit(x, N_peaks, *args),
                         data[:, 0], data[:, 1],
                         # p0= [1]*N_peaks + [0]*N_peaks+[1e-10]*N_peaks,  # 用于告知要拟合的函数的参数个数
                         p0=[.1] * N_peaks * 3,  # 用于告知要拟合的函数的参数个数
                         # bounds= numpy.array([
                         #     [0]*N_peaks + [0]*N_peaks+[-numpy.inf]*N_peaks,
                         #     [numpy.inf] * N_peaks + [numpy.inf] * N_peaks + [numpy.inf] * N_peaks,
                         # ])#.T
                         # ,maxfev = 10000
                         # ,method = 'dogbox'
                         )

    def peaks(self):
        pass

    @staticmethod
    def Gauss_fit_known_mu(data: numpy.ndarray, mus):
        """
        拟合为N个峰的高斯函数
        :return:
        """
        from scipy.optimize import curve_fit
        N_peaks = len(mus)

        return curve_fit(lambda x, *args: GaussPeakSeeker.Gauss_N_peak(
            x, args[:N_peaks], mus, args[N_peaks:]),
                         data[:, 0], data[:, 1],
                         # p0= [1]*N_peaks + [0]*N_peaks+[1e-10]*N_peaks,  # 用于告知要拟合的函数的参数个数
                         p0=[.1] * N_peaks * 2,  # 用于告知要拟合的函数的参数个数
                         # bounds= numpy.array([
                         #     [0]*N_peaks + [0]*N_peaks+[-numpy.inf]*N_peaks,
                         #     [numpy.inf] * N_peaks + [numpy.inf] * N_peaks + [numpy.inf] * N_peaks,
                         # ])#.T
                         # ,maxfev = 10000
                         # ,method = 'dogbox'
                         )


from scipy.signal import find_peaks

# def find_peak():
#     output_power_FD = grd.obs[FD_title]['data']  # unit in GHz
#     freq_extrema_indexes = argrelextrema(output_power_FD.values[:, 1], numpy.greater)
#
#     freq_extremas: numpy.ndarray = output_power_FD.iloc[freq_extrema_indexes].values  # 频谱极大值
#     freq_peak_indexes = list(
#         map(lambda v: numpy.where(freq_extremas[:, 1] == v)[0][0], heapq.nlargest(5, freq_extremas[:, 1])))  # 频谱峰值
#     freq_peaks = freq_extremas[freq_peak_indexes]
#     freq_peaks[:, 0] *= 1e9  # Convert to SI unit
plt.ion()

x = numpy.linspace(0, 150, 1000)
# plt.figure()
# plt.plot(x, Gauss_N_peak(x,[1,1,0.2],[0, 11.7*2, 50],[0.1,1,0.1]))
plt.figure()
plt.plot(x, GaussPeakSeeker._Gauss_N_peak_for_curve_fit(x, 3, 1, 1, 0.2, 0, 11.7 * 2, 50, 0.1, 1, 0.1))

# plt.show()


data = grd_parser.GRD(
    # r"D:\MagicFiles\HPM\12.5GHz\优化2\RSSE_template_20230619_225316_75459072.grd",
    r"D:\MagicFiles\HPM\12.5GHz\优化2\RSSE_template_20230620_144903_14697472.grd",
).obs[
    ' FIELD_POWER S.DA @PORT_RIGHT,FFT-#20.2']['data'].values
data = numpy.vstack([[-1, 0],  # 用于使算法定位到0频率处的峰值
                     data])
N_peaks = 3
fit_res = GaussPeakSeeker.Gauss_fit_N_peak(data, N_peaks)
plt.figure()
plt.plot(*data.T, label='data')
peaks_indexes = find_peaks(data[:, 1])[0]
peaks_indexes =[peaks_indexes[i] for i in   numpy.argsort(data[peaks_indexes,1],)[-1:-10:-1]]
plt.scatter(*data[peaks_indexes].T, label='peak')
plt.plot(data[:, 0], GaussPeakSeeker._Gauss_N_peak_for_curve_fit(data[:, 0], N_peaks, *fit_res[0]), label='fitted')
mus =data[peaks_indexes[:2], 0]
fit_res2 = GaussPeakSeeker.Gauss_fit_known_mu(data,  mus)[0]
plt.plot(data[:, 0], GaussPeakSeeker.Gauss_N_peak(
            data[:, 0], fit_res2[:len(mus)], mus, fit_res2[len(mus):]),
         label='fitted: known $\mu$')

plt.legend()
plt.show()
