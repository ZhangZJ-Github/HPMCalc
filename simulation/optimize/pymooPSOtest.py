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
from pymoo.core.problem import ElementwiseProblem
import shutil
from simulation.optimize.initialize import Initializer

_1 = pymooPSO

if __name__ == '__main__':

    initial_csv_path = r"F:\changeworld\HPMCalc\simulation\template\TTO\Initialize.csv"
    # 取消注释以在Excel中编辑初始条件
    # Initializer.make_new_initial_csv(initial_csv_path, get_hpmsim())
    initializer = Initializer(initial_csv_path)


    class MyProblem(ElementwiseProblem):
        BIG_NUM = 100

        def __init__(self, *args, **kwargs):
            super(MyProblem, self, ).__init__(*args, n_var=len(initializer.initial_df.columns),
                                              n_obj=1,
                                              n_ieq_constr=0,  # TTOParamPreProcessor.N_constraint_le,
                                              # n_constr=len(constraint_ueq),
                                              xl=initializer.lower_bound,
                                              xu=initializer.upper_bound, **kwargs)
            logger.info("======= Optimization start! ========")
            logger.info(self.elementwise)
            logger.info(self.elementwise_runner)

        def bad_res(self, out):
            out['F'] = [self.BIG_NUM] * self.n_obj
            # self._evaluate(1,dict(), 1,2,3,4,a =1,ka =1)

        def _evaluate(self, x, out: dict, *args, **kwargs
                      ):
            hpmsim = get_hpmsim()
            hpmsim.template.copy_template_to_working_dir()
            shutil.copy(initializer.filename,
                        os.path.join(hpmsim.template.working_dir, os.path.split(initializer.filename)[1]))

            # out['G'] = [constr(x) for constr in constraint_ueq]
            # if numpy.any(numpy.array(out['G']) > 0):
            #     logger.warning("并非全部约束条件都满足：%s" % (out['G']))
            #     self.bad_res(out)
            #     return
            logger.info('score = %.2e' %
                        hpmsim.update({initializer.index_to_param_name(i): x[i] for i in
                                       range(len(initializer.initial_df.columns))},
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
            logger.info('SamplingWithGoodEnoughValues.do()')
            global first_sampling

            for j in range(len(res)):
                res[j].X = (res[j].X // initializer.precision).astype(int) * initializer.precision
                logger.info('已按照指定精度截断，res[%d].X = %s' % (j, res[j].X))
            if first_sampling:  # 若为第一次采样则严格按照初始值设置
                for i in range(min(initializer.N_initial, len(res))):
                    res[i].X = initializer.initial_df.loc[i].values
                    logger.info("设置了初始值：res[%d].X = %s\n" % (i, res[i].X,))
                first_sampling = False

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


    algorithm = pymooPSO(
        pop_size=49,
        sampling=SamplingWithGoodEnoughValues(),  # LHS(),
        # ref_dirs=ref_dirs
    )
    # aaaaaaaaa
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
