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

_1 = pymooPSO

# create the reference directions to be used for the optimization
# ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
# ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)

# create the algorithm object
# algorithm = NSGA3(pop_size=92,
#                   ref_dirs=ref_dirs)

# execute the optimization
# problem = get_problem("dtlz4")

# res = minimize(problem,
#                algorithm,
#                seed=1,
#                termination=('n_gen', 600))

# plt.ion()
# Scatter().add(res.F).show()
if __name__ == '__main__':

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
        # lambda params_array: -0.1,
        # lambda params_array: -0.1,
    )


    def obj_func(params: numpy.ndarray, comment=''):
        return -get_hpsim().update(
            {initial_data.columns[i]: params[i] for i in range(len(params))}, comment)


    # score = obj_func(initial_data.loc[0].values, "测试初始值")  # 用初始值测试
    # aaaaaaaaaaaaaaaa

    class MyProblem(ElementwiseProblem):
        BIG_NUM = 100

        def __init__(self, *args, **kwargs):
            super(MyProblem, self, ).__init__(*args, n_var=len(initial_data.columns),
                                              n_obj=1,
                                              n_constr=len(constraint_ueq),
                                              xl=initial_data.loc[1].values,
                                              xu=initial_data.loc[2].values, **kwargs)
            logger.info("===============")
            logger.info(self.elementwise)
            logger.info(self.elementwise_runner)

        def elem_evaluate(self, x, i  # out, *args, **kwargs
                          ):
            hpsim = get_hpsim()
            out = {}
            out['G'] = [constr(x) for constr in constraint_ueq]
            if numpy.any(numpy.array(out['G']) > 0):
                logger.warning("并非全部约束条件都满足：%s" % (out['G']))
                self.bad_res(out)
                return out, i
            logger.info(
                hpsim.update({index_to_param_name(i): x[i] for i in range(len(initial_data.columns))}, 'NSGA-III'))
            try:
                # out['F'] = [
                #     -hpsim.log_df[hpsim.colname_avg_power_score].iloc[-1],
                #     -hpsim.log_df[hpsim.colname_freq_accuracy_score].iloc[-1],
                #     -hpsim.log_df[hpsim.colname_freq_purity_score].iloc[-1]
                # ]

                out['F'] = -hpsim.log_df[hpsim.colname_score]
                logger.info("out['F'] = %s" % out['F'])

            except AttributeError as e:
                logger.warning("忽略的报错：%s" % e)
                self.bad_res(out)
            return out, i

        def bad_res(self, out):
            out['F'] = [self.BIG_NUM] * self.n_obj

        # def _evaluate(self, X, out, *args, **kwargs):
        #     out['F'] = numpy.zeros(( X.shape[0], self.n_obj))
        #     out['G'] = numpy.zeros(( X.shape[0], self.n_constr))
        #
        #     pool = ThreadPoolExecutor(max_workers=7)
        #
        #     def set_out(future: concurrent.futures.Future):
        #         res,i = future.result()
        #         for key in res:
        #             out[key][i] = res[key]
        #     logger.info("X = \n%s"%X)
        #     for i in range(len(X)):
        #         x = X[i]
        #         logger.info("x=%s\ni=%d" % (x,i))
        #         exe = pool.submit(self.elem_evaluate, x,i)
        #         exe.add_done_callback(set_out)
        #     pool.shutdown()
        #     logger.info("out = %s" % out)

        # def _evaluate_vectorized(self, X, out, *args, **kwargs):
        #     logger.info("_evaluate_vectorized")

        def _evaluate(self, x, out, *args, **kwargs
                      ):
            hpsim = get_hpsim()
            out = {}
            out['G'] = [constr(x) for constr in constraint_ueq]
            if numpy.any(numpy.array(out['G']) > 0):
                logger.warning("并非全部约束条件都满足：%s" % (out['G']))
                self.bad_res(out)
                return
            logger.info('score = %.2e' %
                        hpsim.update({index_to_param_name(i): x[i] for i in range(len(initial_data.columns))},
                                     'PSO'))
            try:
                score = hpsim.log_df[hpsim.colname_score]
                out['F'] = [-score * 100]
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
        pop_size=10,
        sampling=SamplingWithGoodEnoughValues(),  # LHS(),
        # ref_dirs=ref_dirs
    )
    # from simulation.optimize.myproblem import MyProblem
    # iter = 0
    # def log_iter():
    #     global  iter
    #     logger.info("迭代至第%d代"%iter)
    res = minimize(
        MyProblem(elementwise_runner=runner
                  ),
        algorithm,
        seed=1,
        termination=('n_gen', 100),
        verbose=True,
        # callback =log_iter,#save_history=True
    )

    # plt.ion()
    Scatter().add(res.F).show()
    aaaaaaaaa
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
