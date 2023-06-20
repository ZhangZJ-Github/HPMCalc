# -*- coding: utf-8 -*-
# @Time    : 2023/6/19 23:16
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : myproblem.py
# @Software: PyCharm


import matplotlib

matplotlib.use('tkagg')
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga3 import NSGA3
from simulation.optimize.optimize_HPM import *
import simulation.task_manager.task
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

rmin_cathode = HPMSim.rmin_cathode

# <=0
constraint_ueq = (
    lambda params_array: params_array[
                             param_name_to_index['%sws1.dz_out%']] - params_array[
                             param_name_to_index['%sws1.dz_in%']],
    lambda params_array: params_array[
                             param_name_to_index['%sws2.dz_out%']] - params_array[
                             param_name_to_index['%sws2.dz_in%']],
    lambda params_array: params_array[
                             param_name_to_index['%sws3.dz_out%']] - params_array[
                             param_name_to_index['%sws3.dz_in%']],
    lambda params_array: params_array[param_name_to_index['%sws1.dz_in%']] - params_array[
        param_name_to_index['%sws1.p%']],
    lambda params_array: params_array[param_name_to_index['%sws2.dz_in%']] - params_array[
        param_name_to_index['%sws2.p%']],
    lambda params_array: params_array[param_name_to_index['%sws3.dz_in%']] - params_array[
        param_name_to_index['%sws3.p%']],

    lambda params_array: (params_array[param_name_to_index['%refcav.rin_right_offset%']] + rmin_cathode) -
                         params_array[
                             param_name_to_index['%refcav.rout%']],

    lambda params_array: (params_array[param_name_to_index['%refcav.rin_right_offset%']] + rmin_cathode) -
                         params_array[param_name_to_index['%sws1.a%']],
    lambda params_array: (params_array[param_name_to_index['%refcav.rin_right_offset%']] + rmin_cathode) -
                         params_array[param_name_to_index['%sws2.a%']],
    lambda params_array: (params_array[param_name_to_index['%refcav.rin_right_offset%']] + rmin_cathode) -
                         params_array[param_name_to_index['%sws3.a%']],
)


def get_hpsim():
    return HPMSim(r"F:\changeworld\HPMCalc\simulation\template\RSSSE\RSSE_template.m2d",
                  r'D:\MagicFiles\HPM\12.5GHz\优化2', 11.7e9, 1e9)


def obj_func(params: numpy.ndarray, comment=''):
    return -get_hpsim().update(
        {initial_data.columns[i]: params[i] for i in range(len(params))}, comment)


from pymoo.core.problem import ElementwiseProblem, Problem


class MyProblem(ElementwiseProblem):
    BIG_NUM = 1

    def __init__(self, *args, **kwargs):
        super(MyProblem, self, ).__init__(*args, n_var=len(initial_data.columns),
                                          n_obj=3,
                                          n_constr=len(constraint_ueq),
                                          xl=initial_data.loc[1],
                                          xu=initial_data.loc[2], **kwargs)

        logger.info(self.elementwise)
        logger.info(self.elementwise_runner)

    def _evaluate(self, x, out, *args, **kwargs):
        hpsim = get_hpsim()
        out['G'] = [constr(x) for constr in constraint_ueq]
        if numpy.any(numpy.array(out['G']) > 0):
            logger.warning("并非全部约束条件都满足：%s" % (out['G']))
            self.bad_res(out)
            return
        logger.info(
            hpsim.update({index_to_param_name(i): x[i] for i in range(len(initial_data.columns))}, 'NSGA-III'))
        try:
            out['F'] = [
                -hpsim.log_df[hpsim.colname_avg_power_score].iloc[-1],
                -hpsim.log_df[hpsim.colname_freq_accuracy_score].iloc[-1],
                -hpsim.log_df[hpsim.colname_freq_purity_score].iloc[-1]
            ]
            logger.info("out['F'] = %s" % out['F'])
        except AttributeError as e:
            logger.warning("忽略的报错：%s" % e)
            self.bad_res(out)

    def bad_res(self, out):
        out['F'] = [self.BIG_NUM] * self.n_obj