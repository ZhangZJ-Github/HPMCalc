# -*- coding: utf-8 -*-
# @Time    : 2024/1/20 12:18
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : CST_test.py
# @Software: PyCharm
"""
比较CST和给定的轴上电场

用于快速检索的关键词：
快速计算CST加速效果
"""
import cst.results
import matplotlib
import numpy
import pandas

from generate_Ez_field6 import Cell, CellChain, Nose

matplotlib.use('tkagg')

import matplotlib.pyplot as plt
from _logging import logger

if __name__ == '__main__':
    plt.ion()

    cellchain = CellChain([
(Cell(-113561949.98682201,Nose(2.1762185438586785,1.8109482018623393),Nose(1.5217192847042633,1.3107333260147371),-0.232866198,0.232866198),18.6808128189687),
(Cell(133315387.54135904,Nose(1.3794301274516583,2.113445026634417),Nose(1.9978967002947698,1.3986733526634776),-0.335920283,0.335920283),28.2495992999687),
(Cell(-146479516.18302366,Nose(1.62226346714395,2.886477749368537),Nose(2.88477780604098,1.6271689656732584),-0.312505797,0.312505797),37.8980253799687),
(Cell(118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),50.22817883006155),
(Cell(-118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),66.34817883006156),
(Cell(118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),82.46817883006156),
(Cell(-118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),98.58817883006155),
(Cell(118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),114.70817883006156),
(Cell(-118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),130.82817883006157),
(Cell(118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),146.94817883006155),
(Cell(-118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),163.06817883006156),
(Cell(118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),179.18817883006156),
(Cell(-118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),195.30817883006156),
(Cell(118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),211.42817883006157),
(Cell(-118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),227.54817883006157),
(Cell(118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),243.66817883006155),
(Cell(-118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),259.78817883006155),
(Cell(118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),275.90817883006156),
(Cell(-118314535.96548162,Nose(1.6067765569554884,2.728040289860505),Nose(2.730376722127042,1.624499901646521),-3.017647653092852,3.017647653092852),292.02817883006156)
])
    proj: cst.results.ProjectFile = cst.results.ProjectFile(
        r'E:\CSTprojects\GeneratorAccelerator\StandingWaveAccelerator_FDSolver.cst', allow_interactive=True)
    res3d: cst.results.ResultModule = proj.get_3d()
    Ez: cst.results.ResultItem = res3d.get_result_item(r'Tables\1D Results\e-field (f=9.46) (1)_Z (Z)')
    Ezdata = numpy.array(Ez.get_data())
    plt.figure()
    __z_normal_cell_start = 80
    mask_normal_cell = Ezdata[:, 0] > __z_normal_cell_start
    Ezdata_normalized = Ezdata[:, 1] / (Ezdata[mask_normal_cell][:, 1]).max()
    plt.plot(*Ezdata.T, label='CST')


    def norm_Ezdata_GPT(z, z0, k):
        Ezdata_GPT_optimized = cellchain.Ez(z - z0)
        return (Ezdata_GPT_optimized)  *k


    from scipy.optimize import curve_fit

    [z0,k], pcov = curve_fit(norm_Ezdata_GPT, *Ezdata.T )
    plt.plot(Ezdata[:, 0],  norm_Ezdata_GPT(Ezdata[:, 0], z0,k), label='GPT, optimized')
    plt.legend()

    flt = (Ezdata[:, 0] > 6) & (Ezdata[:, 0] < 20)

    Ezdata_at_SI_unit = Ezdata.copy()
    Ezdata_at_SI_unit[:, 0] *= 1e-3

    # zs = newEzdata[:, 0]

    from simulation.task_manager.simulator import df_to_gdf
    # Ez_equivalent = Ezdata[:,1]

    from high_capture_efficiency_Ez import GPTEzCaptureSimulator, HighCaptureEffEzTask

    mask_normal_cell =  Ezdata_at_SI_unit[:, 0]> 0.06
    df_to_gdf(pandas.DataFrame({
        'zs': Ezdata_at_SI_unit[:, 0], 'Ez': (lambda Ez:  Ez / max(Ez[mask_normal_cell].max(), - Ez[mask_normal_cell].min()))(Ezdata_at_SI_unit[:, 1])
    }), 'Ez1D.gdf',False)

    GPTEzCaptureSimulator().run_bat('test_acc.bat')



    f = 9.46e9
    fig, axs = plt.subplots(3, 1, sharex=True)
    import scipy.constants as C

    high_capture_task = HighCaptureEffEzTask()

    res = high_capture_task.get_res('')
    traj = high_capture_task.traj

    Ezdata_at_SI_unit = pandas.read_csv('Ez1D.txt', sep='\t').values
    light_line = lambda z: (z - Ezdata_at_SI_unit[:, 0].min()) / C.c
    for parid in traj.dfs:
        axs[0].plot(traj.dfs[parid]['zs'], traj.dfs[parid]['time'] - light_line(traj.dfs[parid]['zs']),alpha = 0.1)
        axs[2].plot(traj.dfs[parid]['zs'], ((traj.dfs[parid]['time']  / (1/f))%0.5)*360,alpha = 0.1)

    axs[1].plot(*Ezdata_at_SI_unit.T)
    axs[1].plot(Ezdata_at_SI_unit[:, 0], numpy.abs(Ezdata_at_SI_unit[:, 1]))
    axs[1].set_xlim(Ezdata_at_SI_unit[:, 0].min(), Ezdata_at_SI_unit[:, 0].max())


    [ax.grid() for ax in axs]

    z_screen = 0.35  # Ezdata_at_SI_unit[-1,0]
    interpdata = traj.interpolate_at_screen(z_screen)
    flter = interpdata['G'] > (6.0e6 / HighCaptureEffEzTask.E0_eV + 1)

    std_E = interpdata[flter]
    plt.figure()
    plt.hist((interpdata['G'] - 1) * HighCaptureEffEzTask.E0_eV / 1e6, bins=500, )
    plt.xlabel('particle energy / MeV')

    logger.info("Capture efficiency: %.3f, avg E: %.2f MeV, std E: %.2f MeV" % (
        res[HighCaptureEffEzTask._Cols.capture_efficiency.name],
        res[HighCaptureEffEzTask._Cols.output_particle_energy_eV_average.name] / 1e6,
        res[HighCaptureEffEzTask._Cols.output_particle_energy_eV_std.name] / 1e6))

