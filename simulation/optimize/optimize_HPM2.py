# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 20:19
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test.py
# @Software: PyCharm
import concurrent.futures
import heapq
import json
import os.path

import matplotlib

import simulation.task_manager.task

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
        rmin_cathode = 18e-3
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
        logger.info("evaluate of SOOptimizer")
        # 以下各项得分最高为1
        weights = {self.colname_avg_power_score: 3,  # TODO: 目前应为奇数
                   self.colname_freq_accuracy_score: 2,  # 频率准确度，如，12.5 GHz的设备输出两个主峰应为0和25 GHz，且幅值接近1:1
                   self.colname_freq_purity_score: 1  # 频率纯度，如，12.5 GHz的设备除了上述两个主峰外，其他频率成分应趋于0
                   }
        freq_peaks = numpy.array(json.loads(res["freq peaks"]))
        freq_accuracy_score = self.freq_accuracy_score(freq_peaks, self.desired_frequency, 1e9)
        avg_power_score = self.avg_power_score(res["avg. power"], self.desired_mean_power)
        freq_purity_score = self.freq_purity_score(freq_peaks)

        res[self.colname_avg_power_score] = avg_power_score
        res[self.colname_freq_accuracy_score] = freq_accuracy_score
        res[self.colname_freq_purity_score] = freq_purity_score

        return (
                avg_power_score ** weights[self.colname_avg_power_score]
                * freq_accuracy_score ** weights[self.colname_freq_accuracy_score]
                * freq_purity_score ** weights[self.colname_freq_purity_score]
        )  # **(1 / sum(list(weights.values())))

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
        freq_extrema_indexes = argrelextrema(output_power_FD.values[:, 1], numpy.greater)

        freq_extremas: numpy.ndarray = output_power_FD.iloc[freq_extrema_indexes].values  # 频谱极大值
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


if __name__ == '__main__':
    lock = Lock()
    hpmsim = HPMSim(r"F:\changeworld\HPMCalc\simulation\template\RSSSE\RSSE_template.m2d",
                    r'D:\MagicFiles\HPM\12.5GHz\优化')

    vars_ = hpmsim.template.get_variables()
    logger.info(
        vars_
    )
    initial_csv = 'Initialize.csv'
    # 取消注释以在Excel中编辑初始条件
    # pandas.DataFrame(columns=list(vars_)).to_csv(initial_csv,index = False)
    # os.system("start %s"%initial_csv)

    initial_data = pandas.read_csv(initial_csv, encoding=simulation.task_manager.task.CSV_ENCODING)
    initial_data = initial_data[initial_data.columns[1:]]  # 去除备注列
    # initial_data = initial_data[initial_data.columns[initial_data.loc[2] > initial_data.loc[1]]]  # 去除上下限完全一致（无需调整）的变量
    init_params = {col: initial_data[col][0] for col in initial_data}
    # aaaa

    index_to_param_name = lambda i: initial_data.columns[i]
    param_name_to_index = {
        initial_data.columns[i]: i for i in range(len(initial_data.columns))
    }


    # # <=0
    # constraint_ueq = (
    #     lambda params_array: params_array[
    #                              param_name_to_index['%sws1.dz_out%']] - params_array[
    #                              param_name_to_index['%sws1.dz_in%']],
    #     lambda params_array: params_array[
    #                              param_name_to_index['%sws2.dz_out%']] - params_array[
    #                              param_name_to_index['%sws2.dz_in%']],
    #     lambda params_array: params_array[
    #                              param_name_to_index['%sws3.dz_out%']] - params_array[
    #                              param_name_to_index['%sws3.dz_in%']],
    #     lambda params_array: params_array[param_name_to_index['%sws1.dz_in%']] - params_array[
    #         param_name_to_index['%sws1.p%']],
    #     lambda params_array: params_array[param_name_to_index['%sws2.dz_in%']] - params_array[
    #         param_name_to_index['%sws2.p%']],
    #     lambda params_array: params_array[param_name_to_index['%sws3.dz_in%']] - params_array[
    #         param_name_to_index['%sws3.p%']],
    #
    # )

    def get_hpsim():
        return HPMSim(r"F:\changeworld\HPMCalc\simulation\template\RSSSE\RSSE_template.m2d",
                      r'D:\MagicFiles\HPM\12.5GHz\优化2', 11.7e9, 1e9)


    def obj_func(params: numpy.ndarray, comment=''):
        return -get_hpsim().update(
            {initial_data.columns[i]: params[i] for i in range(len(params))}, comment)


    # score = obj_func(initial_data.loc[0].values)  # 用初始值测试
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.problems import get_problem
    from pymoo.optimize import minimize

    res = minimize(obj_func,
                   algorithm,
                   ('n_gen', 10),
                   seed=1,
                   verbose=True)

    # calculate a hash to show that all executions end with the same result
    print("hash", res.F.sum())

    from sko.tools import set_run_mode
    from sko.PSO import PSO

    obj_func_with_comment = lambda x: obj_func(x, 'PSO')
    set_run_mode(obj_func_with_comment, 'multithreading')
    pso = PSO(func=obj_func_with_comment,
              n_dim=len(initial_data.columns), pop=30,
              lb=[initial_data[col][1] for col in initial_data.columns],
              ub=[initial_data[col][2] for col in initial_data.columns],
              # constraint_ueq=constraint_ueq
              )
    pso.run(precision=1e-4)
    # aaaaaaaaaa

    # from scipy.optimize import minimize
    #
    # optimize_res = minimize(
    #     obj_func,
    #     initial_data.loc[0].values,
    #     bounds=[
    #         (initial_data[col][1], initial_data[col][2]) for col in initial_data.columns
    #     ],
    #     method='Nelder-Mead'
    # )

    import ADAM

    dx_for_get_partial_diff = initial_data.loc[3].values


    def get_partial_diff_for_ith_param(params, i, old_score):
        params_changed = params.copy()
        params_changed[i] += dx_for_get_partial_diff[i]
        # sim = get_hpsim()
        logger.info("get_partial_diff_for_ith_param\ni = %d" % i)
        return (obj_func(params_changed) - old_score) / dx_for_get_partial_diff[i], i


    from concurrent.futures import ThreadPoolExecutor


    def parallel_get_grad(params: numpy.ndarray, old_score):
        grad = numpy.zeros(params.shape)
        pool = ThreadPoolExecutor(max_workers=7)

        # old_score = obj_func(params)

        def set_grad(future: concurrent.futures.Future):
            res = future.result()
            i = res[1]
            grad[i] = res[0]
            logger.info("i = %d, 更新后 Grad = %s" % (i, grad))

        for i in range(len(params)):
            exe = pool.submit(get_partial_diff_for_ith_param, params, i, old_score)
            exe.add_done_callback(lambda future: set_grad(future))
        pool.shutdown()
        return grad


    def get_partial_func_for_ith_param(params, i, dx_for_get_partial_diff: numpy.ndarray = dx_for_get_partial_diff):
        params_changed = params.copy()
        params_changed[i] += dx_for_get_partial_diff[i]
        # sim = get_hpsim()
        comment = "用于计算偏微分，%s" % (index_to_param_name(i))
        logger.info(comment)
        return obj_func(params_changed, comment), i


    def parallel_get_partial_func(params: numpy.ndarray, dx_for_get_partial_diff):
        partial_func = numpy.zeros(params.shape)  # 目标函数的偏微分
        pool = ThreadPoolExecutor(max_workers=7)

        # old_score = obj_func(params)

        def set_partial_func(future: concurrent.futures.Future):
            res = future.result()
            v, i = res
            partial_func[i] = v
            logger.info("i = %d, 更新后偏微分 = %s" % (i, partial_func))

        for i in range(len(params)):
            exe = pool.submit(get_partial_func_for_ith_param, params, i, dx_for_get_partial_diff)
            exe.add_done_callback(lambda future: set_partial_func(future))
        pool.shutdown()
        return partial_func


    adam = ADAM.Adam(obj_func, initial_data.loc[0].values,
                     parallel_get_partial_func,
                     dx_for_get_partial_diff,
                     lb=initial_data.loc[1].values, ub=initial_data.loc[2].values,
                     unit_steps=initial_data.loc[4].values)
    ADAM.AdamPlot.plot_fig(adam, 0.5)
    # adam.run(dx_for_get_partial_diff)
