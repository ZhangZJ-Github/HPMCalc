# -*- coding: utf-8 -*-
# @Time    : 2023/11/22 17:35
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : Genac10G50keV.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2023/11/3 16:37
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : Genac20G100keV_1.py
# @Software: PyCharm
import os.path
import typing

import filenametool
from _logging import logger

import simulation.optimize.hpm
from simulation.template.GeneratorAccelerator.main import *

initialize_csv = r'initialize-sweepB.csv'
get_initializer = lambda: Initializer(initialize_csv)  # 动态调用，每次生成新个体时都会重新读一遍优化配置，从而支持在运行时临时修改优化配置


class Genac(HPMSimWithInitializer):
    def get_res(self, m2d_path: str, ) -> dict:
        grd = grd_parser.GRD(
            filenametool.ExtTool(os.path.splitext(m2d_path)[0]).get_name_with_ext(filenametool.ExtTool.FileType.grd))
        Bz_particle_line_max = grd.ranges[r' FIELD BZ_ST @LINE_PARTICLE_MOVING$ #2.1'][0]['data'][1].max()
        res = super(Genac, self).get_res(m2d_path, r' FIELD_POWER S.DA @PORT_RIGHT,FFT-#20.1',
                                         r' FIELD_POWER S.DA @PORT_LEFT,FFT-#19.1')
        res['Bz_particle_line_max'] = Bz_particle_line_max
        return res
    def evaluate(self, res: dict):
        """

        :param res:
        :return: 最终评分
        """
        logger.info("evaluate of HPSim")
        # 以下各项得分最高为1
        weights = {
            self.colname_power_eff_score: 2*0,
            self.colname_out_power_score: 3,
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


def get_genac(get_initializer: typing.Callable[[], typing.Union[Initializer, None]] = get_initializer):
    return Genac(get_initializer(), r"F:\papers\OptimizationHPM\ICCEM\support\优化后\sweepB\RSSE_MC_optimized.m2d",
                 r"F:\papers\OptimizationHPM\ICCEM\support\优化后\sweepB",
                 desired_frequency=11.7e9,
                 desired_mean_power=1e9,
                 lock=lock)


import grd_parser

if __name__ == '__main__':
    # Initializer.make_new_initial_csv(initialize_csv,
    #                                  get_genac(lambda: None))
    # genac = get_genac()
    # res = genac.get_res(r"E:\GeneratorAccelerator\Genac\optmz\另行处理\CouplerGap3mm.grd")
    # genac.evaluate(res)
    # res



    get_genac().re_evaluate()

    # job = simulation.optimize.hpm.MaunualJOb(get_initializer(), get_genac)
    # job.run(2)
