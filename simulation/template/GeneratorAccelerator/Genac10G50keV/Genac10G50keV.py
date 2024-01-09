# -*- coding: utf-8 -*-
# @Time    : 2023/11/22 17:35
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : Genac10G50keV.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2023/11/3 16:37
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : Genac20G100keV_1.py
# @Software: PyCharm
import typing

import simulation.optimize.hpm.hpm
from simulation.template.GeneratorAccelerator.main import *

initialize_csv = r'initialize.csv'
get_initializer = lambda: Initializer(initialize_csv)  # 动态调用，每次生成新个体时都会重新读一遍优化配置，从而支持在运行时临时修改优化配置


class Genac(HPMSimWithInitializer):
    def get_res(self, m2d_path: str, ) -> dict:
        et = ExtTool(os.path.splitext(m2d_path)[0])
        grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
        # in_power_TD_titile = r' FIELD_POWER S.DA @LEFT,FFT-#5.1'
        out_power_TD_titile: str = r' FIELD_POWER S.DA @COUPLER.LINE,FFT-#13.1'
        TD_title = out_power_TD_titile
        output_power_TD = grd.obs[TD_title]['data']
        dT_for_period_avg = 3 * 1 / (2 * self.desired_frequency)
        mean_output_power = -self._get_mean(output_power_TD,
                                           dT_for_period_avg)  # 功率：2倍频；为获得比较平滑的结果，这里扩大了采样周期
        mean_input_power = 6e6  # self._get_mean(input_power_TD, dT_for_period_avg)
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


def get_genac(get_initializer: typing.Callable[[], typing.Union[Initializer, None]] = get_initializer):
    return Genac(get_initializer(), r'Genac10G50keV-temp.m2d',
                 r'E:\GeneratorAccelerator\Genac\optmz\Genac10G50keV-1\RoughMesh',
                 desired_frequency=10e9,
                 desired_mean_power=6e6 * 0.6,
                 lock=lock)


import grd_parser

if __name__ == '__main__':
    # Initializer.make_new_initial_csv(initialize_csv,
    #                                  get_genac(lambda: None))
    # genac = get_genac()
    # res = genac.get_res(r"E:\GeneratorAccelerator\Genac\optmz\另行处理\CouplerGap3mm.grd")
    # genac.evaluate(res)
    # res
    # aaa
    optjob = simulation.optimize.hpm.hpm.OptimizeJob(get_initializer(), get_genac)
    optjob.algorithm = PSO(
        pop_size=14,
        sampling=simulation.optimize.hpm.hpm.SamplingWithGoodEnoughValues(optjob.initializer),  # LHS(),
        # ref_dirs=ref_dirs
    )
    optjob.run(n_threads=1)
