# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 17:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : high_capture_efficiency_Ez.py
# @Software: PyCharm
"""
Optimize the $E_z$ field profile to achieve both good capture efficiency and low energy spread.
优化出综合性能最好的纵向电场分布
"""
import matplotlib

matplotlib.use('tkagg')
import enum
from enum import auto
from generate_Ez_field6 import Nose, Cell, CellChain
import numpy
import pandas
import scipy.constants as C
from pymoo.algorithms.soo.nonconvex.pso import PSO

import pygpt
import simulation.task_manager.initialize
# from simulation.optimize.accelerator.LINAC.SW.generate_Ez_field3 import CellChain, Cell
from simulation.optimize.hpm.hpm import HPMSimWithInitializer
from simulation.task_manager.simulator import InputFileTemplateBase
from simulation.task_manager.simulator import df_to_gdf
from simulation.task_manager.task import CachedTask
from _logging import logger
from simulation.post_processing.GPT_trajectory import GPTTraj

initial_csv_path = 'high_capture_efficiency_Ez.initial.csv'
initializer = simulation.task_manager.initialize.Initializer(initial_csv_path)


class EzInputTemplate(InputFileTemplateBase):
    zs_m = numpy.linspace(-20, 310, 2000) * 1e-3

    def __init__(self):
        super(EzInputTemplate, self).__init__('fake.template', '.', )
        self.outputfile = 'Ez1D.gdf'

    def generate_and_to_disk(self, args_to_build_cells: dict):
        # c1z1 = args_to_build_cells['c1.z1']
        c1z2 = args_to_build_cells['c1.z2']
        c1factor = args_to_build_cells['c1.factor']

        # c2z1 = args_to_build_cells['c2.z1']
        c2z2 = args_to_build_cells['c2.z2']
        c2factor = args_to_build_cells['c2.factor']

        # c3z1 = args_to_build_cells['c3.z1']
        c3z2 = args_to_build_cells['c3.z2']
        c3factor = args_to_build_cells['c3.factor']
        c4z2 = 3.017647653092852

        c4z = 50.22817883006155
        # Nsigma = 4
        min_dz = 6 * 1.5

        c3z = c4z - max(args_to_build_cells['c3.dz'], c3z2 + min_dz + c4z2)
        c2z = c3z - max(args_to_build_cells['c2.dz'], c2z2 + min_dz + c3z2)
        c1z = c2z - max(args_to_build_cells['c1.dz'], c1z2 + min_dz + c2z2)

        normal_cell_dz = 16.12  # C.c / 9.3e9 / 2 * 1e3
        # c1_nose2_zsigma = 1.23273844
        # c2_nose1_zsigma = 1.3889799250418073
        # c2_nose2_zsigma = 1.2968551942437123

        cellchain = CellChain([
            (Cell(-39524049.663693234 * c1factor, Nose(2.1762185438586785, 1.8109482018623393),
                  Nose(1.5217192847042633, 1.3107333260147371), -c1z2, c1z2), c1z),
            (Cell(88640861.53084418 * c2factor, Nose(1.3794301274516583, 2.113445026634417),
                  Nose(1.9978967002947698, 1.3986733526634776), -c2z2, c2z2),
             c2z),
            (Cell(-111096163.37660348 * c3factor, Nose(1.62226346714395, 2.886477749368537),
                  Nose(2.88477780604098, 1.6271689656732584), -c3z2, c3z2),
             c3z),

            *[(Cell(118314535.96548162 * (-1) ** i, Nose(1.6067765569554884, 2.728040289860505),
                    Nose(2.730376722127042, 1.624499901646521), -c4z2, c4z2),
               c4z + i * normal_cell_dz) for i in range(16)],
        ])

        Ez = cellchain.Ez(self.zs_m * 1e3)
        mask_normal_cell = self.zs_m > 0.06
        df = pandas.DataFrame(
            {'zs': self.zs_m, 'Ez': Ez / max(Ez[mask_normal_cell].max(), - Ez[mask_normal_cell].min())})

        df_to_gdf(df, self.outputfile, False)
        logger.info(cellchain)
        return self.outputfile


class GPTEzCaptureSimulator(simulation.task_manager.simulator.GeneralParticleTracerSim):
    # def __init__(self):
    #     super(GPTEzCaptureSimulator, self).__init__()
    def run(self, *args, **kwargs):
        return super(GPTEzCaptureSimulator, self).run_bat('test_acc.bat')


class HighCaptureEffEzTask(CachedTask):
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
        self.template: EzInputTemplate = self.template

    def get_res(self, path: str) -> dict:
        traj_gdf = pygpt.gdftomemory('traj.gdf')
        self.traj = traj = GPTTraj(traj_gdf)

        key = 'G'
        get_filter = lambda df: (df['G'] - 1) * self.E0_eV > 4e6

        zout = self.template.zs_m[-1]
        capture_eff = traj.capture_efficiency(0., zout, get_filter)
        gamma_avg = gamma_std = 1
        if capture_eff:
            gamma_avg = traj.average_at_screen(zout, key, get_filter)
            gamma_std = traj.std_at_screen(zout, key, get_filter)
        output_particle_energy_eV_average = (gamma_avg - 1) * self.E0_eV
        Ezmax = 50e6  # 电场幅值，供参考
        weights = {self._Cols.capture_efficiency.name: 2,
                   self._Cols.output_particle_energy_eV_average.name: 1,
                   self._Cols.output_particle_energy_eV_std.name: 1}

        return {
            self._Cols.capture_efficiency.name: capture_eff,
            self._Cols.output_particle_energy_eV_average.name: output_particle_energy_eV_average,
            self._Cols.output_particle_energy_eV_std.name: (gamma_std) * self.E0_eV,
            self.Colname.score: (
                                        capture_eff ** weights[self._Cols.capture_efficiency.name] *

                                        (output_particle_energy_eV_average / (
                                                Ezmax * (self.template.zs_m[-1] - self.template.zs_m[0]))
                                         ) ** weights[self._Cols.output_particle_energy_eV_average.name] *

                                        numpy.exp(-gamma_std / gamma_avg) ** weights[
                                            self._Cols.output_particle_energy_eV_std.name]
                                ) ** (1 / sum(weights.values()))

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
    # job.run(n_threads=1,
    #         copy_template_and_initialize_csv_to_working_dir=False)

    # 查看initialize.csv里的最后一组初始值对应的结果
    import matplotlib.pyplot as plt

    plt.ion()
    job = simulation.optimize.hpm.hpm.MaunualJOb(initializer,
                                                 lambda: HighCaptureEffEzTask())
    job.run(1)

    f = 9.3e9
    fig, axs = plt.subplots(3, 1, sharex=True)
    import scipy.constants as C

    high_capture_task = HighCaptureEffEzTask()

    res = high_capture_task.get_res('')
    traj = high_capture_task.traj

    Ezdata_at_SI_unit = pandas.read_csv('Ez1D.txt', sep='\t').values
    light_line = lambda z: (z - Ezdata_at_SI_unit[:, 0].min()) / C.c
    for parid in traj.dfs:
        axs[0].plot(traj.dfs[parid]['zs'], traj.dfs[parid]['time'] - light_line(traj.dfs[parid]['zs']),alpha = 0.1)
        axs[2].plot(traj.dfs[parid]['zs'], ((traj.dfs[parid]['time']  / (1/f))%1.0)*360,alpha = 0.1)

    axs[1].plot(*Ezdata_at_SI_unit.T)
    axs[1].plot(Ezdata_at_SI_unit[:, 0], numpy.abs(Ezdata_at_SI_unit[:, 1]))
    axs[1].set_xlim(Ezdata_at_SI_unit[:, 0].min(), Ezdata_at_SI_unit[:, 0].max())


    [ax.grid() for ax in axs]

    z_screen = 0.35  # Ezdata_at_SI_unit[-1,0]
    interpdata = traj.interpolate_at_screen(z_screen)
    flter = interpdata['G'] > (6.0e6 / HighCaptureEffEzTask.E0_eV + 1)

    std_E = interpdata[flter]
    plt.figure()
    plt.hist((interpdata['G'] - 1) * HighCaptureEffEzTask.E0_eV / 1e6, bins=100, )
    plt.xlabel('particle energy / MeV')

    logger.info("Capture efficiency: %.3f, avg E: %.2f MeV, std E: %.2f MeV" % (
        res[HighCaptureEffEzTask._Cols.capture_efficiency.name],
        res[HighCaptureEffEzTask._Cols.output_particle_energy_eV_average.name] / 1e6,
        res[HighCaptureEffEzTask._Cols.output_particle_energy_eV_std.name] / 1e6))
