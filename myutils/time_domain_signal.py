# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/11 14:08
@Auth ： Zi-Jing Zhang (张子靖)
@File ：time_domain_signal.py
@IDE ：PyCharm
"""
import numpy
import pandas
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import argrelextrema


class Signal:
    @staticmethod
    def periodic_mean(df: pandas.DataFrame, DeltaT):
        """
        获取近周期的时间序列数据df在时间间隔DeltaT内的均值
        :param df: 第0列为时间，第1列为值
        :param DeltaT:
        :return:
        """
        colname_period = 'period'
        df[colname_period] = df[0] // (DeltaT)
        return df.groupby(colname_period).mean().iloc[:-1]  # 周期平均功率。倒数第一个周期可能不完整，结果波动很大，故不取。
    @staticmethod
    def spectrum(TD_data:numpy.ndarray, consider_peak_at_0_Hz=False) -> numpy.ndarray:
        """
        将时域信号TD_data转化为频谱
        @param TD_data: 时域信号， shape (N, 2), 其中第0列为时刻，第1列为信号值。

        @return: shape (N//2+1, 2), 其中第0列为频率，第1列为幅值
        """

        # outpower_time_seq = TD_data[
        #     (TD_data[0] >= tmin) & (TD_data[0] <= tmax)].values
        sp = numpy.abs((fft(TD_data[:, 1])))[:TD_data.shape[0] // 2]
        freqs = (fftfreq(TD_data[:, 0].shape[-1], TD_data[1, 0] - TD_data[0, 0]))[
                :TD_data.shape[0] // 2]

        # 为了便于后续查找0频率处的峰值，新增一个数据点
        if consider_peak_at_0_Hz:
            freqs = numpy.array([freqs[0] - freqs[1], *freqs])
            sp = numpy.array([0, *sp])


        return numpy.array([freqs, sp]).T
    @staticmethod
    def frequency_of_monochromatic_data(TD_data:numpy.ndarray):
        """
        对于频率单一的数据，采用平均峰值距离的方式计算周期，相比FFT准确性更高
        @param TD_data:
        @return:
        """
        peak_indexes = argrelextrema(TD_data[:,1],numpy.greater)[0]
        times = TD_data[peak_indexes,0]
        return 1/( numpy.diff(times)).mean()




    @staticmethod
    def peaks(spectrum_data: numpy.ndarray):
        """
        查找频谱上的峰值

        @param spectrum_data: 频谱信号，shape (N_freqs, 2)，其中，第0列为频率，第1列为幅值
        @return: shape (N_peaks, 2)， 第0列为频率，第1列为峰高。按照峰高降序。
        """
        freqs, sp = spectrum_data.T

        peak_indexes_sorted_by_frequency_ascending = argrelextrema(sp, numpy.greater)[0]  # 按照频率大小升序的峰值索引

        peaks_sorted_by_frequency_ascending: numpy.ndarray = (numpy.array((freqs, sp)).T)[
                                                             peak_indexes_sorted_by_frequency_ascending,
                                                             :]  # 频谱极大值 第0列：频率， 第1列：幅值

        # 按照峰高降序的峰值索引
        peak_indexes_sorted_by_peak_height_descending = numpy.argsort(peaks_sorted_by_frequency_ascending[:, 1], )[::-1]

        peaks_sorted_by_peak_height_descending = peaks_sorted_by_frequency_ascending[
                                                 peak_indexes_sorted_by_peak_height_descending, :]
        return peaks_sorted_by_peak_height_descending
