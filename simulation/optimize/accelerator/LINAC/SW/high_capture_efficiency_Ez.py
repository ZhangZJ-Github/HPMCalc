# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 17:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : high_capture_efficiency_Ez.py
# @Software: PyCharm
import enum
from enum import auto

import numpy
import pandas
import scipy.constants as C
from pymoo.algorithms.soo.nonconvex.pso import PSO

import pygpt
import simulation.task_manager.initialize
from simulation.optimize.accelerator.LINAC.SW.generate_Ez_field import CavChainEzGenerator
from simulation.optimize.hpm.hpm import HPMSimWithInitializer
from simulation.task_manager.simulator import InputFileTemplateBase
from simulation.task_manager.simulator import df_to_gdf
from simulation.task_manager.task import TaskBase

initializer = simulation.task_manager.initialize.Initializer('initial.csv')
from simulation.post_processing.GPT_trajectory import GPTTraj


class EzInputTemplate(InputFileTemplateBase):
    def __init__(self):
        super(EzInputTemplate, self).__init__('fake.template', '.', )
        self.outputfile = 'Ez1D.gdf'

    def generate_and_to_disk(self, args_to_build_cells: dict):
        cells = CavChainEzGenerator.from_1D_array(list(args_to_build_cells.values())[:-1])

        zs = numpy.linspace(cells.zmids[0] - 2 * cells.cell_chain[0].L, cells.zmids[-1] + cells.cell_chain[-1].L, 1000)
        self.zs = zs
        Ez = cells.Ez(zs)
        Ez /= Ez.max()
        df = pandas.DataFrame({'z': zs, 'Ez': Ez / Ez.max()})

        df_to_gdf(df, self.outputfile)
        return self.outputfile


class GPTEzCaptureSimulator(simulation.task_manager.simulator.GeneralParticleTracerSim):
    # def __init__(self):
    #     super(GPTEzCaptureSimulator, self).__init__()
    def run(self, *args, **kwargs):
        return super(GPTEzCaptureSimulator, self).run_bat('test_acc.bat')


class HighCaptureEffEzTask(TaskBase):
    """
    中间文件名是固定的，因此不支持并行
    且程序运行已足够快，没必要并行
    """
    E0_eV = C.m_e * C.c ** 2 / C.eV  # 电子的静止能量

    class _Cols(enum.Enum):
        """
        中间结果列
        """
        capture_efficiency = auto()
        output_particle_energy_eV_average = auto()
        output_particle_energy_eV_std = auto()

    def __init__(self):
        super(HighCaptureEffEzTask, self).__init__(
            EzInputTemplate(), GPTEzCaptureSimulator()
        )

    def get_res(self, path: str) -> dict:
        traj_gdf = pygpt.gdftomemory('traj.gdf')
        traj = GPTTraj(traj_gdf, 9.3e9, )
        key = 'G'

        gamma_avg = traj.average_at_screen(self.template.zs[-1], key)
        output_particle_energy_eV_average = (gamma_avg - 1) * self.E0_eV
        gamma_std = traj.std_at_screen(self.template.zs[-1], key)

        capture_eff = traj.capture_efficiency()
        Ezmax = 100e6  # 电场幅值，供参考

        return {
            self._Cols.capture_efficiency.name: capture_eff,
            self._Cols.output_particle_energy_eV_average.name: output_particle_energy_eV_average,
            self._Cols.output_particle_energy_eV_std.name: (gamma_std) * self.E0_eV,
            self.Colname.score:
                capture_eff *
                (output_particle_energy_eV_average / (Ezmax * (self.template.zs[-1] - self.template.zs[0]))) *
                numpy.exp(-gamma_std / gamma_avg)
        }

    def evaluate(self, res: dict) -> float:
        return res[self.Colname.score]

    def find_old_res(self, params: dict, precisions: dict = {}) -> str: return ''


if __name__ == '__main__':
    job = simulation.optimize.hpm.hpm.OptimizeJob(initializer,
                                                  lambda: HighCaptureEffEzTask())

    job.algorithm = PSO(
        pop_size=60,
        sampling=simulation.optimize.hpm.hpm.SamplingWithGoodEnoughValues(job.initializer),  # LHS(),
        # ref_dirs=ref_dirs
    )
    job.run(n_threads=1,
            copy_template_and_initialize_csv_to_working_dir=False)

    # 查看initialize.csv里的最后一组初始值对应的结果
    # job = simulation.optimize.hpm.hpm.MaunualJOb(initializer,
    #                                              lambda: HighCaptureEffEzTask())
    # job.run(1)
    # traj_gdf = pygpt.gdftomemory('traj.gdf')
    # traj = GPTTraj(traj_gdf, 9.3e9, )
    # plt.figure()
    # traj.plot_AppleGate_diagram(ax= plt.gca())
    # traj.capture_efficiency()
    #
