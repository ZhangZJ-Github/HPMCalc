# -*- coding: utf-8 -*-
# @Time    : 2023/10/22 21:35
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : main.py
# @Software: PyCharm
import heapq
import json
import os

import grd_parser
import numpy
import pandas
from filenametool import ExtTool
from scipy.signal import argrelextrema

import simulation.optimize.hpm
from simulation.optimize.hpm import HPMSimWithInitializer
from simulation.task_manager.initialize import Initializer

initialize_csv = r'initialize.csv'
get_initializer = lambda: Initializer(initialize_csv)  # 动态调用，每次生成新个体时都会重新读一遍优化配置，从而支持在运行时临时修改优化配置


class Genac(HPMSimWithInitializer):
    def get_res(self, m2d_path: str, ) -> dict:
        et = ExtTool(os.path.splitext(m2d_path)[0])
        grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
        in_power_TD_titile = r' FIELD_POWER S.DA @LEFT,FFT-#4.1'
        out_power_TD_titile: str = r' FIELD_POWER S.DA @RIGHT,FFT-#5.1'
        TD_title = out_power_TD_titile
        output_power_TD, input_power_TD = grd.obs[TD_title]['data'], grd.obs[in_power_TD_titile]['data']
        dT_for_period_avg = 3 * 1 / (2 * self.desired_frequency)
        mean_output_power = self._get_mean(output_power_TD,
                                           dT_for_period_avg)  # 功率：2倍频；为获得比较平滑的结果，这里扩大了采样周期
        mean_input_power = self._get_mean(input_power_TD,
                                          dT_for_period_avg)
        FD_title = TD_title[:-1] + '2'
        output_power_FD = grd.obs[FD_title]['data']  # unit in GHz
        output_power_FD: pandas.DataFrame = pandas.DataFrame(
            numpy.vstack([[-1, 0],  # 用于使算法定位到0频率处的峰值
                          output_power_FD.values]),
            columns=output_power_FD.columns)
        freq_extrema_indexes = argrelextrema(output_power_FD.values[:, 1], numpy.greater)

        freq_extremas: numpy.ndarray = output_power_FD.iloc[freq_extrema_indexes].values  # 频谱极大值
        # from scipy.signal import find_peaks
        # freq_peak_indexes = find_peaks(output_power_FD.values[:, 1],0   )[0]

        # 按照峰高排序，选最高的几个
        freq_peak_indexes = list(
            map(lambda v: numpy.where(freq_extremas[:, 1] == v)[0][0], heapq.nlargest(5, freq_extremas[:, 1])))  # 频谱峰值
        freq_peaks = freq_extremas[freq_peak_indexes]
        freq_peaks[:, 0] *= 1e9  # Convert to SI unit
        res = {
            self.colname_avg_power_in: mean_input_power,  # TODO
            self.colname_avg_power_out: mean_output_power,
            self.colname_freq_peaks: json.dumps(freq_peaks.tolist())
        }
        return res

    def evaluate(self, res: dict):
        super(Genac, self).evaluate(res)
        res[self.colname_power_eff_score] = eff = res[self.colname_avg_power_out] / res[self.colname_avg_power_in]
        return eff


from threading import Lock

lock = Lock()


def get_genac():
    return Genac(get_initializer(), r'template.m2d', r"F:\11-11\10g-optimize2", 25e9,
                 lock=lock)


if __name__ == '__main__':
    # Initializer.make_new_initial_csv(initialize_csv,
    #     Genac(None,r'template.m2d', r"F:\11-11\10g-optimize2", 25e9, ))
    # genac = get_genac()
    # res = genac.get_res(r"E:\GeneratorAccelerator\Genac\optmz\另行处理\CouplerGap3mm.grd")
    # genac.evaluate(res)
    # res
    # aaa
    optjob = simulation.optimize.hpm.hpm.MaunualJOb(get_initializer(), get_genac)
    # optjob.algorithm = PSO(
    #     pop_size=21,
    #     sampling=simulation.optimize.hpm.SamplingWithGoodEnoughValues(optjob.initializer),  # LHS(),
    #     # ref_dirs=ref_dirs
    # )
    optjob.run()
