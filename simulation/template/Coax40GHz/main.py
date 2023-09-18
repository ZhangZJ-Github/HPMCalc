# -*- coding: utf-8 -*-
# @Time    : 2023/9/15 15:55
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : main.py
# @Software: PyCharm
import os.path

import grd_parser
import total_parser

import simulation.optimize.hpm as hpm
import simulation.optimize.initialize


class Coax40GHzSim(hpm.HPMSimWithInitializer):
    class Coax40GHzSimRawColname:
        Ipeak, Iavg = 'Ipeak, Iavg'.split(', ')  # 电流峰值，电流平均值

        pass

    class Coax40GHzSimEvaluatingeColname:
        modulated_depth = 'modulated_depth'  # 电流峰值/电流平均值
        pass

    def get_res(self, m2d_path: str, ) -> dict:
        res = {}
        obs_emitted_I_titles: str = r' EMITTED ELECTRON CURRENT @EMIT_ZONE-#1.1'
        obs_I_titles = []

        grd = grd_parser.GRD(
            total_parser.ExtTool(os.path.splitext(m2d_path)[0]).get_name_with_ext(total_parser.ExtTool.FileType.grd))
        for key in grd.obs.keys():
            if key.startswith(r' FIELD_INTEGRAL J_ELECTRON.DA'):
                obs_I_titles.append(key)
        dt_for_get_avg = 2 * 1. / self.desired_frequency
        res[self.Coax40GHzSimRawColname.Iavg] = I_avg = self._get_mean(grd.obs[obs_emitted_I_titles]['data'],
                                                                       dt_for_get_avg)
        res[self.Coax40GHzSimRawColname.Ipeak] = I_peak = min(
            [grd.obs[title]['data'][1].min() for title in obs_I_titles])
        return res

    def evaluate(self, res: dict):
        modulated_depth = res[self.Coax40GHzSimRawColname.Ipeak] / res[self.Coax40GHzSimRawColname.Iavg]
        res[self.Coax40GHzSimEvaluatingeColname.modulated_depth] = modulated_depth
        return modulated_depth


initial_csv = r"F:\changeworld\HPMCalc\simulation\template\Coax40GHz\Initialize.csv"
initializer = simulation.optimize.initialize.Initializer(initial_csv)


def get_coax40GHzsim():
    return Coax40GHzSim(initializer,
                        r'F:\changeworld\HPMCalc\simulation\template\Coax40GHz\40GHzTemp.m2d',
                        r'E:\HPM\40GHz\optimize', 40e9, 3e9, lock=hpm.lock)


if __name__ == '__main__':
    # initializer.make_new_initial_csv(initial_csv,
    #                                  get_coax40GHzsim())
    # coax40GHz_sim = get_coax40GHzsim()
    # res = coax40GHz_sim.get_res(r"E:\HPM\40GHz\optimize\40_ghz-2.m2d")
    # logger.info(coax40GHz_sim.evaluate(res))
    manualjob = hpm.MaunualJOb(initializer, get_coax40GHzsim)
    manualjob.run()
