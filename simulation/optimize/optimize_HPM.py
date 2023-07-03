# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 20:19
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test.py
# @Software: PyCharm
import heapq
import json
import os.path

import matplotlib

matplotlib.use('tkagg')

import grd_parser
import pandas
from _logging import logger
from total_parser import ExtTool
from scipy.signal import argrelextrema
from simulation.task_manager.task import TaskBase
import numpy
from threading import Lock


class HPMSim(TaskBase):
    rmin_cathode = 18e-3
    # 评分明细
    EVAL_COLUMNS = colname_avg_power_score, colname_freq_accuracy_score, colname_freq_purity_score = [
        "avg_power_score",
        "freq_accuracy_score",  # 频率准确度，如，12.5 GHz的设备输出两个主峰应为0和25 GHz，且幅值接近1:1
        "freq_purity_score"  # 频率纯度，如，12.5 GHz的设备除了上述两个主峰外，其他频率成分应趋于0
    ]

    RAW_RES_COLUMNS = colname_avg_power, colname_freq_peaks = [
        "avg. power",
        "freq peaks"
    ]

    def __init__(self, template_name, working_dir=r'D:\MagicFiles\HPM\12.5GHz\优化', desired_frequency=12.5e9,
                 desired_mean_power=1e9, lock: Lock = Lock()):
        super(HPMSim, self).__init__(template_name, working_dir, lock)
        # self.log_df = self.log_df.reindex(self.log_df.columns.tolist()+self.EVAL_COLUMNS)
        # self.log_df.columns += (self.RAW_RES_COLUMNS+ self.EVAL_COLUMNS )
        self.desired_frequency = desired_frequency
        self.desired_mean_power = desired_mean_power

    def params_check(self, params: dict) -> bool:
        """
        将参数合法化
        :param params:
        :return:
        """
        rmin_cathode = self.rmin_cathode
        params['%sws1.dz_in%'] = max(params['%sws1.dz_in%'], params['%sws1.dz_out%'])
        params['%sws2.dz_in%'] = max(params['%sws2.dz_in%'], params['%sws2.dz_out%'])
        params['%sws3.dz_in%'] = max(params['%sws3.dz_in%'], params['%sws3.dz_out%'])
        params['%sws1.p%'] = max(params['%sws1.dz_in%'], params['%sws1.p%'])
        params['%sws2.p%'] = max(params['%sws2.dz_in%'], params['%sws2.p%'])
        params['%sws3.p%'] = max(params['%sws3.dz_in%'], params['%sws3.p%'])
        params["%sws1.N%"] = int(params["%sws1.N%"])
        params["%sws2.N%"] = int(params["%sws2.N%"])
        params["%sws3.N%"] = int(params["%sws3.N%"])

        params['%refcav.rin_right_offset%'] = min(params['%refcav.rin_right_offset%'],
                                                  params['%refcav.rout%'] - rmin_cathode)

        params['%sws1.a%'] = max(params['%sws1.a%'], rmin_cathode + params["%refcav.rin_right_offset%"])
        params['%sws2.a%'] = max(params['%sws2.a%'], rmin_cathode + params["%refcav.rin_right_offset%"])
        params['%sws3.a%'] = max(params['%sws3.a%'], rmin_cathode + params["%refcav.rin_right_offset%"])

        return True

    def evaluate(self, res: dict):
        """

        :param res:
        :return: 最终评分
        """
        logger.info("evaluate of HPSim")
        # 以下各项得分最高为1
        weights = {self.colname_avg_power_score: 3,  # TODO: 目前应为奇数
                   self.colname_freq_accuracy_score: 2,  # 频率准确度，如，12.5 GHz的设备输出两个主峰应为0和25 GHz，且幅值接近1:1
                   self.colname_freq_purity_score: 1  # 频率纯度，如，12.5 GHz的设备除了上述两个主峰外，其他频率成分应趋于0
                   }
        freq_peaks = numpy.array(json.loads(res["freq peaks"]))
        freq_accuracy_score = self.freq_accuracy_score(freq_peaks, self.desired_frequency, .5e9)
        avg_power_score = self.avg_power_score(res["avg. power"], self.desired_mean_power)
        freq_purity_score = self.freq_purity_score(freq_peaks)

        res[self.colname_avg_power_score] = avg_power_score
        res[self.colname_freq_accuracy_score] = freq_accuracy_score
        res[self.colname_freq_purity_score] = freq_purity_score
        score = (
                avg_power_score ** weights[self.colname_avg_power_score]
                * freq_accuracy_score ** weights[self.colname_freq_accuracy_score]
                * freq_purity_score ** weights[self.colname_freq_purity_score]
        )
        return numpy.sign(score) * numpy.abs(score) ** (1 / sum(list(weights.values())))

    @staticmethod
    def freq_purity_score(freq_peaks: numpy.ndarray, ):
        ratio = freq_peaks[2, 1] / freq_peaks[1, 1]  # 杂峰 / 第二主峰
        return 1 - ratio

    def get_res(self, m2d_path: str) -> dict:
        et = ExtTool(os.path.splitext(m2d_path)[0])
        grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
        TD_title = ' FIELD_POWER S.DA @PORT_RIGHT,FFT-#20.1'
        output_power_TD = grd.obs[TD_title]['data']
        colname_period = 'period'
        output_power_TD[colname_period] = output_power_TD[0] // (
                3 * 1 / (2 * self.desired_frequency))  # 功率：2倍频；为获得比较平滑的结果，这里扩大了采样周期
        mean_output_power = output_power_TD.groupby(colname_period).mean().iloc[-2][
            1]  # 倒数第二个周期的平均功率。倒数第一个周期可能不全，结果波动很大，故不取。
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
            self.colname_avg_power: mean_output_power,
            self.colname_freq_peaks: json.dumps(freq_peaks.tolist())
        }
        return res

    @staticmethod
    def avg_power_score(avg_power, desired_power):
        return 2 * (1 / (1 + numpy.exp(
            -(3 * avg_power) / desired_power)) - 0.5)  # sigmoid函数，平均输出为0则0分，正无穷为1分，负无穷为-1分

    @staticmethod
    def freq_accuracy_score(freq_peaks_sorted_by_magn: numpy.ndarray, desired_freq, freq_tol):
        """
        :param freq_peaks_sorted_by_magn:
        :param desired_freq:
        :param freq_tol:
        :return: 满分：1
        """
        g = lambda x, mu, sigma: numpy.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        max_2_freq_peaks = freq_peaks_sorted_by_magn[:2]
        max_2_freq_peaks_sorted_by_freq = max_2_freq_peaks[max_2_freq_peaks[:, 0].argsort()]
        desired_freq_SDA = numpy.array([0, desired_freq * 2])

        return ((numpy.array([
            g(max_2_freq_peaks_sorted_by_freq[:, 0], desired_freq_SDA[i], freq_tol).sum()
            for i in range(len(desired_freq_SDA))]).sum()
                 ) / (2 + g(desired_freq_SDA[0], desired_freq_SDA[1], freq_tol) * 2) +
                + numpy.exp(
                    - (max_2_freq_peaks_sorted_by_freq[0, 1] / max_2_freq_peaks_sorted_by_freq[1, 1] - 1) ** 2 / (
                            2 * 0.2 ** 2)
                )
                ) / 2


lock = Lock()


def get_hpsim(lock=lock):
    return HPMSim(r"F:\changeworld\HPMCalc\simulation\template\RSSSE\RSSE_template.m2d",
                  r'D:\MagicFiles\HPM\12.5GHz\优化6', 11.7e9, 1e9, lock=lock)


if __name__ == '__main__':
    avg_power_score = HPMSim.avg_power_score(0.5e9, 1e9)
