# -*- coding: utf-8 -*-
# @Time    : 2023/10/10 16:31
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : main.py
# @Software: PyCharm

import os.path

import grd_parser
import total_parser

import simulation.optimize.hpm as hpm
import simulation.task_manager.initialize


class MySim(hpm.HPMSimWithInitializer):
    class MySimRawColname:
        Ezmax, Ezmin, Pout_in_GW = 'Ezmax, Ezmin, Pout_in_GW'.split(', ')

    class MySimEvaluatingeColname:
        Ezm = 'Ezm_in_MV/m'  # 峰谷差/2

    def get_res(self, m2d_path: str, ) -> dict:
        res = {}
        obs_Ez_in_extractor1_title: str = r' FIELD E1 @EXTRACTOR1.OBSPT,FFT-#10.1'
        obs_SDA_title = r' FIELD_POWER S.DA @RIGHT,FFT-#4.1'
        # obs_I_titles = []

        grd = grd_parser.GRD(
            total_parser.ExtTool(os.path.splitext(m2d_path)[0]).get_name_with_ext(total_parser.ExtTool.FileType.grd))
        # for key in grd.obs.keys():
        #     if key.startswith(r' FIELD_INTEGRAL J_ELECTRON.DA'):
        #         obs_I_titles.append(key)
        dt_for_get_avg = 2 * 1. / self.desired_frequency
        res[self.MySimRawColname.Ezmax] = Ezmax = self._get_max(
            grd.obs[obs_Ez_in_extractor1_title]['data'],
            dt_for_get_avg)
        res[self.MySimRawColname.Ezmin] = Ezmin = self._get_min(
            grd.obs[obs_Ez_in_extractor1_title]['data'],
            dt_for_get_avg)
        res[self.MySimRawColname.Pout_in_GW] = self._get_mean(grd.obs[obs_SDA_title]['data'], dt_for_get_avg) / 1e9
        return res

    def evaluate(self, res: dict):
        Ezm = (res[self.MySimRawColname.Ezmax] - res[
            self.MySimRawColname.Ezmin]) / 2 / 1e6
        res[self.MySimEvaluatingeColname.Ezm] = Ezm
        return Ezm


initial_csv = r"initialize.csv"
initializer = simulation.task_manager.initialize.Initializer(initial_csv)


def get_simobj():
    return MySim(initializer,
                 r'base.m2d',
                 r'E:\HPM\40GHz\TWT\ChangeDriverBunchTsigma', 40e9, 3e9, lock=hpm.lock)


if __name__ == '__main__':
    # simulation.optimize.initialize.Initializer.make_new_initial_csv(initial_csv,
    #                                  get_simobj())
    # coax40GHz_sim = get_simobj()
    # res = coax40GHz_sim.get_res(r"E:\HPM\40GHz\TWT\temp\bunch_extractor_base_freq-noslot-change_Ipeak.m2d")
    # logger.info(coax40GHz_sim.evaluate(res))
    # aaa

    manualjob = hpm.MaunualJOb(initializer, get_simobj)
    manualjob.run()
