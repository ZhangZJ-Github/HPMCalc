# -*- coding: utf-8 -*-
# @Time    : 2023/6/19 21:00
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : pymootest.py
# @Software: PyCharm

import matplotlib

matplotlib.use('tkagg')
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.soo.nonconvex.pso import PSO as pymooPSO
from simulation.optimize.optimize_HPM import *
import simulation.task_manager.task
from pymoo.core.problem import ElementwiseProblem
import shutil

_1 = pymooPSO

if __name__ == '__main__':

    initial_csv = r"D:\zhangzijing\codes\hpmcalc\simulation\template\CS\Initialize - 副本.csv"
    # 取消注释以在Excel中编辑初始条件
#     _df = pandas.DataFrame(columns=list(get_hpsim().template.get_variables()))
#     row_names = """备注
# 初始值
# 下界
# 上界
# 计算偏微分的步长
# 单位步长
# """.split('\n')
#     _df = pandas.concat([pandas.DataFrame({row_names[0]: row_names[1:]}), _df])
#
#     _df.to_csv(initial_csv, index=False, encoding=simulation.task_manager.task.CSV_ENCODING)
#     os.system("start %s" % initial_csv)

    # aaaaaa

    initial_data = pandas.read_csv(initial_csv, encoding=simulation.task_manager.task.CSV_ENCODING)
    initial_data = initial_data[initial_data.columns[1:]]  # 去除备注列
    # initial_data = initial_data[initial_data.columns[initial_data.loc[2] > initial_data.loc[1]]]  # 去除上下限完全一致（无需调整）的变量
    init_params = {col: initial_data[col][0] for col in initial_data}
    # aaaa


    index_to_param_name = lambda i: initial_data.columns[i]
    param_name_to_index = {
        initial_data.columns[i]: i for i in range(len(initial_data.columns))
    }
    get_hpsim().update(init_params) # 初始值测试
    aaaaaaaaaaa


    rmin_cathode = HPMSim.rmin_cathode

    # <=0
    constraint_ueq = (
        # lambda params_array: params_array[
        #                          param_name_to_index['%sws1.dz_out%']] - params_array[
        #                          param_name_to_index['%sws1.dz_in%']],
        # lambda params_array: params_array[
        #                          param_name_to_index['%sws2.dz_out%']] - params_array[
        #                          param_name_to_index['%sws2.dz_in%']],
        # lambda params_array: params_array[
        #                          param_name_to_index['%sws3.dz_out%']] - params_array[
        #                          param_name_to_index['%sws3.dz_in%']],
        # lambda params_array: params_array[param_name_to_index['%sws1.dz_in%']] - params_array[
        #     param_name_to_index['%sws1.p%']],
        # lambda params_array: params_array[param_name_to_index['%sws2.dz_in%']] - params_array[
        #     param_name_to_index['%sws2.p%']],
        # lambda params_array: params_array[param_name_to_index['%sws3.dz_in%']] - params_array[
        #     param_name_to_index['%sws3.p%']],
        #
        # lambda params_array: (params_array[param_name_to_index['%refcav.rin_right_offset%']] + rmin_cathode) -
        #                      params_array[
        #                          param_name_to_index['%refcav.rout%']],
        #
        # lambda params_array: (params_array[param_name_to_index['%refcav.rin_right_offset%']] + rmin_cathode) -
        #                      params_array[param_name_to_index['%sws1.a%']],
        # lambda params_array: (params_array[param_name_to_index['%refcav.rin_right_offset%']] + rmin_cathode) -
        #                      params_array[param_name_to_index['%sws2.a%']],
        # lambda params_array: (params_array[param_name_to_index['%refcav.rin_right_offset%']] + rmin_cathode) -
        #                      params_array[param_name_to_index['%sws3.a%']],
    )


    class MyProblem(ElementwiseProblem):
        BIG_NUM = 100

        def __init__(self, *args, **kwargs):
            super(MyProblem, self, ).__init__(*args, n_var=len(initial_data.columns),
                                              n_obj=1,
                                              n_ieq_constr=len(constraint_ueq),
                                              # n_constr=len(constraint_ueq),
                                              xl=initial_data.loc[1].values,
                                              xu=initial_data.loc[2].values, **kwargs)
            logger.info("===============")
            logger.info(self.elementwise)
            logger.info(self.elementwise_runner)

        def bad_res(self, out):
            out['F'] = [self.BIG_NUM] * self.n_obj
            # self._evaluate(1,dict(), 1,2,3,4,a =1,ka =1)

        def _evaluate(self, x, out: dict, *args, **kwargs
                      ):
            hpsim = get_hpsim()
            hpsim.template.copy_template_to_working_dir()
            shutil.copy(initial_csv, os.path.join(hpsim.template.working_dir, os.path.split(initial_csv)[1]))

            out['G'] = [constr(x) for constr in constraint_ueq]
            if numpy.any(numpy.array(out['G']) > 0):
                logger.warning("并非全部约束条件都满足：%s" % (out['G']))
                self.bad_res(out)
                return
            logger.info('score = %.2e' %
                        hpsim.update({index_to_param_name(i): x[i] for i in range(len(initial_data.columns))},
                                     'PSO'))
            try:
                score = hpsim.log_df[hpsim.colname_score][0]
                out['F'] = [-score * self.BIG_NUM]

                #     [
                #     -hpsim.log_df[hpsim.colname_avg_power_score].iloc[-1],
                #     -hpsim.log_df[hpsim.colname_freq_accuracy_score].iloc[-1],
                #     -hpsim.log_df[hpsim.colname_freq_purity_score].iloc[-1]
                # ]
                logger.info("out['F'] = %s" % out['F'])

            except AttributeError as e:
                logger.warning("忽略的报错：%s" % e)
                self.bad_res(out)
            return




    from multiprocessing.pool import ThreadPool

    from pymoo.core.problem import StarmapParallelization

    n_threads = 7
    pool = ThreadPool(n_threads)
    # n_proccess = 8
    # pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)
    # ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    from pymoo.operators.sampling.lhs import LHS

    first_sampling = True


    class SamplingWithGoodEnoughValues(LHS):
        """
        给定初始值的采样
        可以提前指定足够好的结果
        """

        def __init__(self):
            super(SamplingWithGoodEnoughValues, self).__init__()

        def do(self, problem, n_samples, **kwargs):
            res = super(SamplingWithGoodEnoughValues, self).do(problem, n_samples, **kwargs)
            global first_sampling

            if first_sampling:
                res[0].X = initial_data.loc[0].values
                first_sampling = False
                logger.info("设置了初始值：res[0].X = %s\n此后first_sampling = %s" % (res[0].X, first_sampling))
            for i in range(len(res)):
                while True:
                    G = [constr(res[i].x) for constr in constraint_ueq]
                    if numpy.any(numpy.array(G) > 0):
                        logger.warning("（Sampling）并非全部约束条件都满足for %d：%s\n重新采样……" % (i, G))
                        res[i].X = super(SamplingWithGoodEnoughValues, self).do(problem, 1, **kwargs)[0].X
                        continue
                    else:
                        break
            return res


    algorithm = pymooPSO(
        pop_size=50,
        sampling=SamplingWithGoodEnoughValues(),  # LHS(),
        # ref_dirs=ref_dirs
    )
    res = minimize(
        MyProblem(elementwise_runner=runner
                  ),
        algorithm,
        seed=1,
        termination=('n_gen', 100),
        verbose=True,
        # callback =log_iter,#save_history=True
    )
    Scatter().add(res.F).show()
