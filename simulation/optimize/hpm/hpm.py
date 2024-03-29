# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 20:19
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test.py
# @Software: PyCharm
import heapq
import json
import os.path
import typing

import matplotlib

matplotlib.use('tkagg')

from pymoo.core.problem import ElementwiseProblem

from multiprocessing.pool import ThreadPool

from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.soo.nonconvex.pso import PSO
import simulation.task_manager.initialize

from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import StarmapParallelization

from concurrent.futures import ThreadPoolExecutor
import grd_parser
import pandas

from _logging import logger
from total_parser import ExtTool

# TODO: 2023年10月16日00:47:49 此处import失败：
# Process finished with exit code -1073741819 (0xC0000005)
from scipy.signal import argrelextrema

from simulation.task_manager.task import MAGICTaskBase
import numpy
from threading import Lock
import shutil
from pymoo.optimize import minimize


class HPMSim(MAGICTaskBase):
    # 评分明细
    EVAL_COLUMNS = colname_power_eff_score, colname_out_power_score, colname_freq_accuracy_score, colname_freq_purity_score = [
        "power efficiency score",
        "out power score",
        "freq accuracy score",  # 频率准确度，如，12.5 GHz的设备输出两个主峰应为0和25 GHz，且幅值接近1:1
        "freq purity score"  # 频率纯度，如，12.5 GHz的设备除了上述两个主峰外，其他频率成分应趋于0
    ]

    RAW_RES_COLUMNS = colname_avg_power_in, colname_avg_power_out, colname_freq_peaks = [
        "avg. power in",
        "avg. power out",
        "freq peaks"
    ]

    class ResKeys:
        avg_power_in = 'avg. power in'
        avg_power_out = 'avg. power out'
        freq_peaks = 'freq peaks'

    def __init__(self, template_name, working_dir=r'"E:\RBWO1\一段结构模板', desired_frequency=40e9,
                 desired_mean_power=1e9, lock: Lock = Lock()):
        super(HPMSim, self).__init__(template_name, working_dir, lock)
        # self.log_df = self.log_df.reindex(self.log_df.columns.tolist()+self.EVAL_COLUMNS)
        # self.log_df.columns += (self.RAW_RES_COLUMNS+ self.EVAL_COLUMNS )
        self.desired_frequency = desired_frequency
        self.desired_mean_power = desired_mean_power

    def params_check(self, params: dict) -> bool:
        # Do nothing
        return super(HPMSim, self).params_check(params)

    def evaluate(self, res: dict):
        """

        :param res:
        :return: 最终评分
        """
        logger.info("evaluate of HPSim")
        # 以下各项得分最高为1
        weights = {
            self.colname_power_eff_score: 2,
            self.colname_out_power_score: 1,
            self.colname_freq_accuracy_score: 2,  # 频率准确度，如，12.5 GHz的设备输出两个主峰应为0和25 GHz，且幅值接近1:1
            self.colname_freq_purity_score: 1  # 频率纯度，如，12.5 GHz的设备除了上述两个主峰外，其他频率成分应趋于0
        }
        freq_peaks = numpy.array(json.loads(res[self.ResKeys.freq_peaks]))

        res[self.colname_out_power_score] = avg_power_score = self.avg_power_score(res[self.ResKeys.avg_power_out],
                                                                                   self.desired_mean_power)
        res[self.colname_freq_accuracy_score] = freq_accuracy_score = self.freq_accuracy_score(freq_peaks,
                                                                                               self.desired_frequency,
                                                                                               .5e9)
        res[self.colname_freq_purity_score] = freq_purity_score = self.freq_purity_score(freq_peaks)
        res[self.colname_power_eff_score] = power_eff_score = self.power_efficiency_score(
            res[self.colname_avg_power_in], res[self.colname_avg_power_out])
        score = (
                numpy.abs(power_eff_score
                          ) **
                weights[self.colname_power_eff_score] *
                avg_power_score ** weights[self.colname_out_power_score]
                * freq_accuracy_score ** weights[self.colname_freq_accuracy_score]
                * freq_purity_score ** weights[self.colname_freq_purity_score]
        )
        return numpy.sign(score) * numpy.abs(score) ** (1 / sum(list(weights.values())))

    @staticmethod
    def freq_purity_score(freq_peaks: numpy.ndarray, ):
        ratio = freq_peaks[2, 1] / freq_peaks[1, 1]  # 杂峰 / 第二主峰
        return 1 - ratio

    @staticmethod
    def _get_mean(df: pandas.DataFrame, DeltaT):
        """
        获取近周期的时间序列数据df在时间间隔DeltaT内的均值
        :param df: 第0列为时间，第1列为值
        :param DeltaT:
        :return:
        """
        colname_period = 'period'
        df[colname_period] = df[0] // (DeltaT)
        return df.groupby(colname_period).mean().iloc[-2][1]  # 倒数第二个周期的平均功率。倒数第一个周期可能不全，结果波动很大，故不取。

    @staticmethod
    def _get_min(df: pandas.DataFrame, DeltaT):
        """
        获取近周期的时间序列数据df在时间间隔DeltaT内的最小值
        :param df: 第0列为时间，第1列为值
        :param DeltaT:
        :return:
        """
        colname_period = 'period'
        df[colname_period] = df[0] // (DeltaT)
        return df.groupby(colname_period).min().iloc[-2][1]  # 倒数第二个周期的平均功率。倒数第一个周期可能不全，结果波动很大，故不取。

    @staticmethod
    def _get_max(df: pandas.DataFrame, DeltaT):
        """
        获取近周期的时间序列数据df在时间间隔DeltaT内的最小值
        :param df: 第0列为时间，第1列为值
        :param DeltaT:
        :return:
        """
        colname_period = 'period'
        df[colname_period] = df[0] // (DeltaT)
        return df.groupby(colname_period).max().iloc[-2][1]  # 倒数第二个周期的平均功率。倒数第一个周期可能不全，结果波动很大，故不取。

    def get_res(self, m2d_path: str, out_power_TD_titile: str = r' FIELD_POWER S.DA @RIGHT,FFT-#4.1',
                in_power_TD_titile: str = r" FIELD_POWER S.DA @LEFT,FFT-#3.1", ) -> dict:
        """
        从结果文件中提取出关心的参数，供下一步分析
        :param m2d_path:
        :param out_power_TD_titile:
        :return:
        """
        et = ExtTool(os.path.splitext(m2d_path)[0])
        grd = grd_parser.GRD(et.get_name_with_ext(ExtTool.FileType.grd))
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

    @staticmethod
    def avg_power_score(avg_power, desired_power):
        return 2 * (1 / (1 + numpy.exp(
            -(3 * avg_power) / desired_power)) - 0.5)  # sigmoid函数的变形。平均输出为0则0分，正无穷为1分，负无穷为-1分

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
                 ) / (2 + g(desired_freq_SDA[0], desired_freq_SDA[1], freq_tol) * 2) * numpy.exp(
            - (max_2_freq_peaks_sorted_by_freq[0, 1] / max_2_freq_peaks_sorted_by_freq[1, 1] - 1) ** 2 / (
                    2 * 0.2 ** 2)
        )
                ) ** (1 / 2)
        # return (g(max_2_freq_peaks_sorted_by_freq[0, 0], desired_freq_SDA[0], freq_tol) * g(
        #     max_2_freq_peaks_sorted_by_freq[1, 0], desired_freq_SDA[0], freq_tol) * numpy.exp(
        #     - (max_2_freq_peaks_sorted_by_freq[0, 1] / max_2_freq_peaks_sorted_by_freq[1, 1] - 1) ** 2 / (
        #             2 * 0.2 ** 2)
        # )
        #         ) ** (1 / 2)

    @staticmethod
    def power_efficiency_score(avg_power_in, avg_power_out, ):
        return avg_power_out / avg_power_in


lock = Lock()


def get_hpmsim(lock=lock):
    return HPMSim(r"E:\RBWO1\一段结构模板\40ghz_2.m2d",
                  r"E:\ref_test", 40e9, 3e9, lock=lock)


class HPMSimWithInitializer(HPMSim):
    """
    具有基本的参数检查功能的HPMSim子类。
    参数检查功能依赖于initialize中写入的规则，主要包括越界判断、按指定精度截断。
    pymoo模块支持越界判断功能，因此没必要在这里重复判断。
    若要实现更复杂的参数检查规则，可以继承此类（建议相关代码文件置于对应的template文件夹），override params_check方法。
    """

    def __init__(self, initializer: simulation.task_manager.initialize.Initializer, *args, **kwargs):
        super(HPMSimWithInitializer, self).__init__(*args, **kwargs)
        self.initializer = initializer

    def params_check(self, params: dict) -> bool:
        return super(HPMSimWithInitializer, self).params_check(params)

    def find_old_res(self, params: dict) -> str:
        return super(HPMSimWithInitializer, self).find_old_res(params, self.initializer.precision_df.to_dict())


class SamplingWithGoodEnoughValues(LHS):
    """
    给定初始值的采样
    可以提前指定足够好的结果
    """

    def __init__(self, initializer: simulation.task_manager.initialize.Initializer):
        super(SamplingWithGoodEnoughValues, self).__init__()
        self.initializer = initializer

    def do(self, problem, n_samples, **kwargs):
        # 只在每个个体的第一代执行
        res = super(SamplingWithGoodEnoughValues, self).do(problem, n_samples, **kwargs)
        logger.info('SamplingWithGoodEnoughValues.do()')
        logger.info('len(res) = %d' % len(res))

        for i in range(min(self.initializer.N_initial, len(res))):
            res[i].X = self.initializer.initial_df.loc[i].values
            logger.info("设置了初始值：res[%d].X = %s\n" % (i, res[i].X,))
            # first_sampling = False

        # for i in range(len(res)):
        #     while True:
        #         G = [constr(res[i].x) for constr in constraint_ueq]
        #         if numpy.any(numpy.array(G) > 0):
        #             logger.warning("（Sampling）并非全部约束条件都满足for %d：%s\n重新采样……" % (i, G))
        #             res[i].X = super(SamplingWithGoodEnoughValues, self).do(problem, 1, **kwargs)[0].X
        #             continue
        #         else:
        #             break
        return res


def get_HPMSimWithInitializerExample():
    return HPMSimWithInitializer(
        simulation.task_manager.initialize.Initializer(r"/simulation/template/TTO/Initialize.csv"),
        r"F:\changeworld\HPMCalc\simulation\template\TTO\TTO-template.m2d",
        r'E:\HPM\11.7GHz\optimize\TTO\from_good_5', 11.7e9, 3e9, lock=lock)


class MyProblem(ElementwiseProblem):
    BIG_NUM = 1

    def __init__(self, initializer: simulation.task_manager.initialize.Initializer,
                 method_to_get_hpm: typing.Callable[[], HPMSim],
                 *args, **kwargs,
                 ):

        super(MyProblem, self, ).__init__(*args, n_var=len(initializer.initial_df.columns),
                                          n_obj=1,
                                          n_ieq_constr=0,  # TTOParamPreProcessor.N_constraint_le,
                                          # n_constr=len(constraint_ueq),
                                          xl=initializer.lower_bound,
                                          xu=initializer.upper_bound, **kwargs)
        logger.info("======= Optimization start! ========")
        self.initializer = initializer
        self.method_to_get_hpm = method_to_get_hpm
        logger.info(self.elementwise)
        logger.info(self.elementwise_runner)
        self.copy_template_and_initialize_csv_to_working_dir = kwargs.get(
            'copy_template_and_initialize_csv_to_working_dir', True)

    def bad_res(self, out):
        out['F'] = [self.BIG_NUM] * self.n_obj
        # self._evaluate(1,dict(), 1,2,3,4,a =1,ka =1)

    def _evaluate(self, x, out: dict, *args, **kwargs
                  ):
        hpmsim = self.method_to_get_hpm()
        if self.copy_template_and_initialize_csv_to_working_dir:
            hpmsim.template.copy_template_to_working_dir()
            shutil.copy(self.initializer.filename,
                        os.path.join(hpmsim.template.working_dir, os.path.split(self.initializer.filename)[1]))

        # out['G'] = [constr(x) for constr in constraint_ueq]
        # if numpy.any(numpy.array(out['G']) > 0):
        #     logger.warning("并非全部约束条件都满足：%s" % (out['G']))
        #     self.bad_res(out)
        #     return
        logger.info('score = %.2e' %
                    hpmsim.update({self.initializer.index_to_param_name(i): x[i] for i in
                                   range(len(self.initializer.initial_df.columns))},
                                  'PSO'))
        try:
            score = hpmsim.log_df[hpmsim.Colname.score][0]
            out['F'] = [-score * self.BIG_NUM]

            #     [
            #     -hpsim.log_df[hpsim.colname_avg_power_score].iloc[-1],
            #     -hpsim.log_df[hpsim.colname_freq_accuracy_score].iloc[-1],
            #     -hpsim.log_df[hpsim.colname_freq_purity_score].iloc[-1]
            # ]
            logger.info("out['F'] = %s" % out['F'])

        except AttributeError as e:
            logger.warning("（来自%s）忽略的报错：%s" % (hpmsim.last_generated_m2d_path, e))
            self.bad_res(out)
        return


from simulation.task_manager.task import LoggedTask


class JobBase:
    def __init__(self, initializer: simulation.task_manager.initialize.Initializer,
                 method_to_get_HPMSimWithInitializer_object: typing.Callable[[], LoggedTask]):
        """
                :param initializer:
                :param method_to_get_HPMSimWithInitializer_object:
                获取新HPMSimWithInitializer对象的方法，即调用method_to_get_HPMSimWithInitializer_object()可以返回一个新的HPMSimWithInitializer对象
                """
        self.initializer = initializer
        self.method_to_get_HPMSimWithInitializer_object = method_to_get_HPMSimWithInitializer_object


class OptimizeJob(JobBase):
    def __init__(self, initializer: simulation.task_manager.initialize.Initializer,
                 method_to_get_HPMSimWithInitializer_object: typing.Callable[[], LoggedTask],
                 ):
        super(OptimizeJob, self).__init__(initializer, method_to_get_HPMSimWithInitializer_object)
        # first_sampling = True
        self.algorithm = PSO(
            pop_size=49,
            sampling=(SamplingWithGoodEnoughValues(self.initializer) if self.initializer.N_initial else LHS()),
            # ref_dirs=ref_dirs
        )

    def run(self, n_threads=7, copy_template_and_initialize_csv_to_working_dir=True):
        pool = ThreadPool(n_threads)
        runner = StarmapParallelization(pool.starmap)
        res = minimize(
            MyProblem(self.initializer, method_to_get_hpm=self.method_to_get_HPMSimWithInitializer_object,
                      elementwise_runner=runner,
                      copy_template_and_initialize_csv_to_working_dir=copy_template_and_initialize_csv_to_working_dir
                      ),
            self.algorithm,
            seed=1,
            termination=('n_gen', 100),
            verbose=True,
            # callback =log_iter,#save_history=True
        )
        Scatter().add(res.F).show()


class MaunualJOb(JobBase):
    def __init__(self, *args, **kwargs):
        super(MaunualJOb, self).__init__(*args, **kwargs)

    def run(self, maxworkers=6):
        with ThreadPoolExecutor(max_workers=maxworkers) as pool:
            run = lambda i: self.method_to_get_HPMSimWithInitializer_object().update(self.initializer.init_params[i])
            for i in range(self.initializer.N_initial):
                pool.submit(run, i)


if __name__ == '__main__':
    # get_hpsim().re_evaluate()
    # scores = HPMSim.avg_power_score(0.5e9, 1e9), HPMSim.freq_accuracy_score(numpy.array([[0, 1e6], [14e9, 1e5]]), 40e9,
    #                                                                         .1e9)
    # HPMSim(r"F:\changeworld\HPMCalc\simulation\template\CS\CS.m2d",
    #        r'E:\HPM\11.7GHz\optimize\CS.1', 11.7e9, 1e9, lock=lock)

    hpmsim = get_hpmsim()
    # res = hpmsim.get_res(r'E:\HPM\11.7GHz\optimize\TTO\from_good\TTO-template_20230813_033810_9.grd', )
    # res = hpmsim.get_res(r'E:\HPM\11.7GHz\optimize\CS.manual\CS_20230719_194902_80508928.m2d', )
    # hpmsim.re_evaluate()
    # score = hpmsim.evaluate(res)
