# -*- coding: utf-8 -*-
# @Time    : 2024/10/12 16:23
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : rf_compressor.py
# @Software: PyCharm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')
import theory.rfCompressor.time_dependent_output as tdo

import pandas
import os
import time
import typing
from multiprocessing.pool import ThreadPool
from threading import Lock

import numpy
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import StarmapParallelization, ElementwiseProblem
from pymoo.operators.sampling.lhs import LHS

from pymoo.optimize import minimize

import simulation
from simulation.task_manager.initialize import Initializer
from simulation.task_manager.task import LoggedTask
from simulation.optimize.hpm.hpm import SamplingWithGoodEnoughValues, JobBase
import cst.interface
import cst.results
from scipy.interpolate import LinearNDInterpolator
from _logging import logger

plt.ion()


def run_cst_history(cst_proj_de: cst.interface.DesignEnvironment, history_list_item: str):
    command = "\n".join(['Sub Main()',
                         history_list_item,
                         # 'RebuildOnParametricChange(False, True)',
                         'End Sub'])
    logger.info(command)
    return cst_proj_de.schematic.execute_vba_code(command, timeout=10)


def set_parameter(cst_proj_de: cst.interface.DesignEnvironment, parameters: dict):
    return run_cst_history(cst_proj_de,
                           "\n".join([*['StoreParameter("%s", %s)' % (key, parameters[key]) for key in parameters],
                                      'RebuildOnParametricChange(False, True)', ]))


class CST_Handler:
    def __init__(self, cst_proj_path, ):
        self.cst_proj_path = cst_proj_path
        # self.de = de = cst.interface.DesignEnvironment(mode=cst.interface.DesignEnvironment.StartMode.Existing)
        # self.cst_proj_de = de.get_open_project(cst_proj_path)
        self.restart_count = -1
        self.start_de()

    def build_paramcombination_df(cst_handler):

        run_ids: list = cst_handler.cst_proj_result.get_3d().get_all_run_ids()
        run_ids.remove(0)
        d = {}
        key_runid = "run_id"
        for run_id in run_ids:
            d[key_runid] = d.get(key_runid, []) + [run_id]
            paramcomb = cst_handler.cst_proj_result.get_3d().get_parameter_combination(run_id)
            for key in paramcomb:
                d[key] = d.get(key, []) + [paramcomb[key]]
        df = pandas.DataFrame(d)
        return df

    def run_cst_history(self, history_list_item: str):
        return run_cst_history(self.cst_proj_de, history_list_item)

    def set_parameter(self, parameters: dict):
        return set_parameter(self.cst_proj_de, parameters)

    def run_solver(self):
        return self.cst_proj_de.modeler.run_solver(timeout=10)

    def start_solver(self):
        return self.cst_proj_de.modeler.start_solver(timeout=10)

    def start_de(self,  # mode = cst.interface.DesignEnvironment.StartMode.New
                 ):
        logger.info("connect to CST")
        try:
            de = cst.interface.DesignEnvironment(mode=cst.interface.DesignEnvironment.StartMode.Existing)
        except RuntimeError as e:
            logger.warning(e)
            de = cst.interface.DesignEnvironment(mode=cst.interface.DesignEnvironment.StartMode.New)
        self.de = de
        self.cst_proj_de = de.open_project(self.cst_proj_path)
        self.cst_proj_result: cst.results.ProjectFile = cst.results.ProjectFile(self.cst_proj_path,
                                                                                allow_interactive=True)
        self.restart_count += 1

    def solver_is_running(self):
        logger.info("check solver_is_running")
        ret = self.cst_proj_de.modeler.is_solver_running(timeout=3)
        return ret


initialize_csv = r'initialize.csv'
get_initializer = lambda: Initializer(initialize_csv)  # 动态调用，每次生成新个体时都会重新读一遍优化配置，从而支持在运行时临时修改优化配置


class TwoCSTSimulationAddress:
    def __init__(self, path_of_simulation_with_GDT: str, run_id_of_simulation_with_GDT: int,
                 path_of_simulation_without_GDT: str, run_id_of_simulation_without_GDT: int,
                 ):
        self.path_of_simulation_with_GDT, self.run_id_of_simulation_with_GDT, self.path_of_simulation_without_GDT, self.run_id_of_simulation_without_GDT = path_of_simulation_with_GDT, run_id_of_simulation_with_GDT, path_of_simulation_without_GDT, run_id_of_simulation_without_GDT

    def to_string(self):
        return "%s*%d\n%s*%d" % (
            self.path_of_simulation_with_GDT, self.run_id_of_simulation_with_GDT, self.path_of_simulation_without_GDT,
            self.run_id_of_simulation_without_GDT)

    def __str__(self):
        return self.to_string()

    @staticmethod
    def from_string(address_string: str):
        strings = address_string.splitlines()

        def f(s: str):
            path, runid = s.split("*")
            return path, int(runid)

        return TwoCSTSimulationAddress(*f(strings[0]), *f(strings[1]))


class RfCompressorOptimizationTask(LoggedTask):
    def __init__(self, cst_handler_with_GDT, cst_handler_without_GDT,
                 # project_path_with_GDT: str, project_path_without_GDT: str = None,
                 lock: Lock = Lock(),
                 initializer: Initializer = None, log_file_name='RF_Compressor.log.csv'):
        # super().__init__(lock, initializer, log_file_name)
        super().__init__(lock, initializer, log_file_name)
        self.cst_handler_with_GDT: CST_Handler = cst_handler_with_GDT  # CST_Handler(project_path_with_GDT)
        # if not project_path_without_GDT: project_path_without_GDT = project_path_with_GDT[:-len(".cst")] + '.ES.cst'
        self.cst_handler_without_GDT: CST_Handler = cst_handler_without_GDT  # CST_Handler(project_path_without_GDT)
        self.old_result: dict = None
        self.restart_de_count = -1

        # self.rerun_count = 0

    def evaluate(self, res: dict):
        return [res["TMPG"], -res["Eabs_max_inside_GDT"]]

    def __find_parameter_in_paramcombination_df(self, params_df: pandas.DataFrame, CST_paramcomb_df: pandas.DataFrame):
        index = numpy.abs(CST_paramcomb_df[params_df.columns] - params_df.values) < self.initializer.precision_df[
            params_df.columns]
        data = CST_paramcomb_df[numpy.all(index, axis=1)].iloc[0]
        logger.info("最接近的记录：\n%s" % (data))

        return data

    def find_run_id(rfc, params: dict):
        CST_paramcomb_df_withGDT = rfc.cst_handler_with_GDT.build_paramcombination_df()
        CST_paramcomb_df_withoutGDT = rfc.cst_handler_without_GDT.build_paramcombination_df()
        param_df = pandas.DataFrame([params])
        data_with_GDT = rfc.__find_parameter_in_paramcombination_df(param_df, CST_paramcomb_df_withGDT)
        data_without_GDT = rfc.__find_parameter_in_paramcombination_df(param_df, CST_paramcomb_df_withoutGDT)
        key_runid = "run_id"
        return data_with_GDT[key_runid], data_without_GDT[key_runid]

    def run(self, param_set: dict) -> str:
        self.find_old_res(param_set)
        if self.old_result: return ""
        try:
            # if 1 :return ""
            if self.restart_de_count > 3:
                logger.info("Too many reruns, so skip")
                return ""
            self.cst_handler_with_GDT.set_parameter(param_set)
            self.cst_handler_without_GDT.set_parameter(param_set)
            logger.info("here1")
            self.cst_handler_with_GDT.start_solver()
            self.cst_handler_without_GDT.start_solver()
            logger.info("here2")

            time.sleep(10)
            while self.cst_handler_with_GDT.solver_is_running() or self.cst_handler_without_GDT.solver_is_running():
                time.sleep(2)
            logger.info("CST simulations done")
            return TwoCSTSimulationAddress(self.cst_handler_with_GDT.cst_proj_path, 0,
                                           self.cst_handler_without_GDT.cst_proj_path, 0).to_string()
        except (RuntimeError, TimeoutError) as e:
            # self.rerun_count += 1
            logger.warning(e)
            logger.info("re-run count = %d" % (self.restart_de_count))
            self.cst_handler_with_GDT.start_de()
            logger.info("here3")

            self.cst_handler_without_GDT.start_de()
            logger.info("here4")

            return self.run(param_set)

    def __get_Eabs_max(self, run_id_withoutGDT):
        """

        :param df_Eabs: CST export 3D field,
        形如
   x [mm]  y [mm]  z [mm]  ...  EyIm [V/m]  EzRe [V/m]  EzIm [V/m]
0       0 -13.456   0.434  ...    0.804892    0.943423   -0.822297
1       0  -9.852   0.434  ...    0.178642   -0.004622    0.003096
2       0  -6.248   0.434  ...    1.015808    2.993860   -2.582654
        :return:
        """
        # run_id_withoutGDT = 0
        df_E_filed = pandas.read_csv(
            os.path.join(os.path.splitext(self.cst_handler_without_GDT.cst_proj_result.filename)[0],
                         r'Export\3d\e-field (f=f0) [3].txt'),
            sep=r'\s\s+',
            skiprows=lambda
                idx: idx == 1, engine='python')
        colname_z = 'z [mm]'
        colname_y = 'y [mm]'
        colname_Eabs = '|E|'
        colnames_E = []
        for col in df_E_filed.columns:
            if col.startswith("E"):
                colnames_E.append(col)

        df_E_filed[colname_Eabs] = Eabs = (df_E_filed[colnames_E] ** 2).sum(axis=1) ** 0.5

        df_E_filed_interpolator = LinearNDInterpolator(df_E_filed[[colname_z, colname_y]].values,
                                                       df_E_filed[colname_Eabs],
                                                       fill_value=0.0
                                                       )
        Eabs_max = Eabs.max()
        parameters_withoutGDT = self.cst_handler_without_GDT.cst_proj_result.get_3d().get_parameter_combination(
            run_id_withoutGDT)
        GDT_r = parameters_withoutGDT['GDT_r']
        GDT_y = parameters_withoutGDT['GDT_y']
        GDT_z = parameters_withoutGDT['GDT_z']
        pts_to_calc_tube_max = numpy.array(
            (numpy.meshgrid(numpy.linspace(-GDT_r, GDT_r, 20), numpy.linspace(-GDT_r, GDT_r, 20)))).transpose(
            (1, 2, 0)).reshape((-1, 2))
        pts_to_calc_tube_max = pts_to_calc_tube_max[
                                   pts_to_calc_tube_max[:, 0] ** 2 + pts_to_calc_tube_max[:, 1] ** 2 <= GDT_r ** 2] + [
                                   GDT_z, GDT_y]
        Eabs_max_inside_GDT = df_E_filed_interpolator(pts_to_calc_tube_max).max()
        # Z,Y = numpy.meshgrid(numpy.arange(260, 300, 0.2),numpy.arange(10,40,0.2))
        # plt.figure()
        # plt.contourf(Z,Y, df_E_filed_interpolator(Z,Y))
        # plt.scatter(*pts_to_calc_tube_max.T)
        return Eabs_max_inside_GDT, Eabs_max,

    def find_old_res(self, params: dict) -> pandas.DataFrame:
        """

        :param params:
        :return: 若没找到，则返回None
        """
        if not os.path.exists(self.log_file_name): return
        log_df = self.load_log()
        columns = self.initializer.init_params_df.columns
        delta = log_df[columns] - pandas.DataFrame([params], )[columns].values
        old_result = log_df[numpy.all(numpy.abs(delta) < self.initializer.precision, axis=1)]
        if len(old_result):
            logger.info("Old result found:\n%s" % (old_result.iloc[0]))
            self.old_result = old_result.iloc[0].to_dict()

    def get_res(self, address_of_cst_simulation: str, **kwargs) -> dict:
        """

        :param address_of_cst_simulation: like "path/to/cst/file.cst*1\npath/to/cst/file.ES.cst*4",
        which means the simulation with GDT is simulation with run_id = 1 of the cst project file "path/to/cst/file.cst",
        and the simulation without GDT is simulation with run_id = 4 of the cst project file "path/to/cst/file.ES.cst",
        :return:
        """
        if self.old_result is not None:
            old_result = self.old_result
            self.old_result = None
            return old_result
        address = TwoCSTSimulationAddress.from_string(address_of_cst_simulation)
        # res =  MyObjectives(TwoCSTSimulationAddress.from_string(address_of_cst_simulation)).to_dict()
        resampled_f = numpy.arange(9e9, 9.6e9, 0.01e9)
        run_id_withGDT = address.run_id_of_simulation_with_GDT
        run_id_withoutGDT = address.run_id_of_simulation_without_GDT
        compressor = tdo.Compressor(
            tdo.build_network_from_CST_S_data(
                tdo.get_S_parameter_from_CST_proj(self.cst_handler_with_GDT.cst_proj_result, "S3,3", run_id_withGDT),
                tdo.get_S_parameter_from_CST_proj(self.cst_handler_with_GDT.cst_proj_result, "S2,3", run_id_withGDT),
            ).interpolate(resampled_f).extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2))),
            tdo.build_network_from_CST_S_data(
                tdo.get_S_parameter_from_CST_proj(self.cst_handler_without_GDT.cst_proj_result, "S3,3",
                                                  run_id_withoutGDT),
                tdo.get_S_parameter_from_CST_proj(self.cst_handler_without_GDT.cst_proj_result, "S2,3",
                                                  run_id_withoutGDT),
            ).interpolate(resampled_f).extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2))),
        )

        f_ref = 9.3e9
        dt = 1 / f_ref / 5.
        ts_to_calculate_TD_response = numpy.arange(-20e-9, 0, dt)
        initial_signal = (ts_to_calculate_TD_response, numpy.sin(2 * numpy.pi * f_ref * ts_to_calculate_TD_response))

        gamma_3 = numpy.array(self.cst_handler_with_GDT.cst_proj_result.get_3d().get_result_item(
            '1D Results\\Port Information\\Gamma\\3(1)', run_id_withGDT).get_data())
        compressor_deembedded = compressor.embed_with_waveguide(gamma_3, -((300 - 40 - 20)) * 1e-3)
        Dt_MESS = compressor_deembedded.correct_Dt_MESS(f_ref, 10e-9)
        df_i3_charging, df_i3_discharging, df_o33, df_o23 = compressor_deembedded.run(initial_signal, 10e-9, Dt_MESS)
        power_extraction_efficiency_of_minimum_compressor = (df_o23[tdo.key_complex][df_o23[0] > 0].abs() ** 2).max()
        G_cav = 1 / (1 - numpy.abs(compressor_deembedded.nw_charging.interpolate([f_ref]).s[0, 0, 0]) ** 2)
        TMPG = G_cav * power_extraction_efficiency_of_minimum_compressor
        S23_abs_withGDT = numpy.abs(compressor.nw_discharging.interpolate([f_ref]).s[0, 1, 0])
        Eabs_max_inside_GDT, Eabs_max = self.__get_Eabs_max(run_id_withoutGDT)
        return {
            "power_extraction_efficiency_of_minimum_compressor": power_extraction_efficiency_of_minimum_compressor,
            "G_cav": G_cav,
            "TMPG": TMPG,
            "S23_abs_withGDT": S23_abs_withGDT,
            "Eabs_max_inside_GDT": Eabs_max_inside_GDT,
            "Eabs_max": Eabs_max,
            "I_PC": (Eabs_max / Eabs_max_inside_GDT) ** 2,
            # self.Colname.score: (TMPG , -Eabs_max_inside_GDT)
        }


class MyProblem(ElementwiseProblem):
    BIG_NUM = 1

    def __init__(self, initializer: simulation.task_manager.initialize.Initializer,
                 method_to_get_hpm: typing.Callable[[], RfCompressorOptimizationTask],
                 *args, **kwargs,
                 ):
        super(MyProblem, self, ).__init__(*args, n_var=len(initializer.initial_df.columns),
                                          n_obj=2,
                                          n_ieq_constr=0,  # TTOParamPreProcessor.N_constraint_le,
                                          # n_constr=len(constraint_ueq),
                                          xl=initializer.lower_bound,
                                          xu=initializer.upper_bound, **kwargs)
        logger.info("======= Optimization start! ========")
        self.initializer = initializer
        self.method_to_get_hpm = method_to_get_hpm
        logger.info(self.elementwise)
        logger.info(self.elementwise_runner)
        # self.copy_template_and_initialize_csv_to_working_dir = kwargs.get(
        #     'copy_template_and_initialize_csv_to_working_dir', True)

    def bad_res(self, out):
        out['F'] = [self.BIG_NUM] * self.n_obj
        # self._evaluate(1,dict(), 1,2,3,4,a =1,ka =1)

    def _evaluate(self, x, out: dict, *args, **kwargs
                  ):
        hpmsim = self.method_to_get_hpm()

        # out['G'] = [constr(x) for constr in constraint_ueq]
        # if numpy.any(numpy.array(out['G']) > 0):
        #     logger.warning("并非全部约束条件都满足：%s" % (out['G']))
        #     self.bad_res(out)
        #     return

        algorithm = kwargs["algorithm"]
        # dict_algorithm = {
        #     "algorithm.n_gen": algorithm.n_gen,
        #     "algorithm.w": algorithm.w,
        #     "algorithm.c1": algorithm.c1,
        #     "algorithm.c2": algorithm.c2,
        #     "algorithm.seed": algorithm.seed,
        # }
        dict_algorithm = {
            "algorithm.n_gen": algorithm.n_gen,
            # "algorithm.w": algorithm.w,
            # "algorithm.c1": algorithm.c1,
            # "algorithm.c2": algorithm.c2,
            "algorithm.seed": algorithm.seed,
        }
        logger.info('score = %s' %
                    hpmsim.update({self.initializer.index_to_param_name(i): x[i] for i in
                                   range(len(self.initializer.initial_df.columns))},
                                  'PSO', dict_algorithm))

        score = hpmsim.log_df[hpmsim.Colname.score][0]

        out['F'] = -numpy.array(score)
        logger.info("out['F'] = %s" % out['F'])

        return


class OptimizeJob(JobBase):
    def __init__(self, initializer: simulation.task_manager.initialize.Initializer,
                 method_to_get_HPMSimWithInitializer_object: typing.Callable[[], LoggedTask],
                 ):
        super(OptimizeJob, self).__init__(initializer, method_to_get_HPMSimWithInitializer_object)
        # first_sampling = True
        # self.algorithm = PSO(
        #     pop_size=25,
        #     sampling=(SamplingWithGoodEnoughValues(self.initializer) if self.initializer.N_initial else LHS()),
        #     # ref_dirs=ref_dirs
        # )
        self.algorithm = NSGA2(
            pop_size=25,
            sampling=(SamplingWithGoodEnoughValues(self.initializer) if self.initializer.N_initial else LHS()),
            # ref_dirs=ref_dirs
        )

    def run(self, n_threads=7, copy_template_and_initialize_csv_to_working_dir=True):
        pool = ThreadPool(n_threads)
        runner = StarmapParallelization(pool.starmap)
        res = minimize(
            MyProblem(self.initializer,
                      method_to_get_hpm=self.method_to_get_HPMSimWithInitializer_object,
                      elementwise_runner=runner,
                      # copy_template_and_initialize_csv_to_working_dir=copy_template_and_initialize_csv_to_working_dir
                      ),
            self.algorithm,
            seed=1,
            termination=('n_gen', 100),
            verbose=True,
            # callback =log_iter,#save_history=True
        )
        return res


if __name__ == '__main__':
    # task = RfCompressorOptimizationTask(r"E:\CSTprojects\rfCompressor\cascadedHT\SES_allCST.cst",
    #                                     initializer=Initializer(initialize_csv))
    # # task.run({ "GDT_z":286,"GDT_y":21})
    # task.get_res("")
    # a = TwoCSTSimulationAddress.from_string("path/to/cst/file.cst*1\npath/to/cst/file.ES.cst*4")

    # task = RfCompressorOptimizationTask(initializer=Initializer(initialize_csv))
    project_path_with_GDT = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_allCST.cst"
    cst_handler_with_GDT = CST_Handler(project_path_with_GDT)
    project_path_without_GDT = project_path_with_GDT[:-len(".cst")] + '.ES.cst'
    cst_handler_without_GDT = CST_Handler(project_path_without_GDT)

    job = OptimizeJob(Initializer(initialize_csv), lambda: RfCompressorOptimizationTask(
        cst_handler_with_GDT, cst_handler_without_GDT,
        initializer=Initializer(initialize_csv), log_file_name="RF_Compressor.log.csv"))
    # rfc :RfCompressorOptimizationTask= job.method_to_get_HPMSimWithInitializer_object()
    # rfc.find_run_id(pandas.DataFrame(data =[ numpy.fromstring("285.8	22.5	18.71	20	17.9	27.9	17.4",sep = ' ')],columns =  rfc.initializer.init_params_df.columns ).iloc[0].to_dict())

    job.run(1, False)
