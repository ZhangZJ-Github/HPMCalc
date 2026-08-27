# -*- coding: utf-8 -*-
# @Time    : 2026/4/1 15:06
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : dose_rate.py
# @Software: PyCharm
import numpy
from scipy.interpolate import interp1d

class CommonUsedUnit:
    def __init__(self):
        self.Gy = 1 #J/kg
        #  1rad是指1g受照射物质吸收任何一种射线100erg（尔格）辐射能时的剂量。 从1986年起用 SI导出单位 Gy（戈〔瑞〕）取代。 1rad=10-2Gy。
        self.Rad = 1e-2* self.Gy

class DoseRateCalculator:
    def __init__(self):
        n_data = numpy.array([
            [3, 6],  # Ek in MeV
            [3, 2.7],  # n
        ]
        )
        self.n_interp = interp1d(*n_data, bounds_error=False, fill_value=tuple(n_data[1]))

    # @staticmethod
    def dose_rate_easy(self,
            Ek_MeV, I_avg_uA):
        """
        Refs: Eqn. (3) in
        Lee Y, Kim S, Kim G, Lee J, Kim I S, Kim J, Shin K Y, Seol Y, Oh T, An N, Lee J, Hwang J, Oh Y, Kang Y. Medical X‐band linear accelerator for high‐precision radiotherapy[J]. Medical Physics, 2021, 48(9): 5327-5342.

        :param Ek_MeV:
        :param I_avg_uA:
        :return: unit in cGy/min
        """
        return 0.067 * I_avg_uA * Ek_MeV **self.n_interp(Ek_MeV)

    def dose_rate_LiuFochengPRApp2025(self,
            Ek_MeV, I_avg_A,d,
                                      a = None):
        """
        Refs: [1] Liu F, Zhang L, Shi J, Zha H, Zhu Y, Gao Q, Zhang F, Hu A, Qiu R, Li J, Huang W, Tang C, Chen H, Yan Q, Liu G, Zhang X, He Y, Liu Y, Liu J, Qiu J, Han Y, Wang J, Wang C, Guo C, Men K, Zhu H, Deng X, Wang W, Hu K. MAX-FLASH: A compact multiangle x-ray system for clinical translation of FLASH radiotherapy[J]. Physical Review Applied, 2025, 24(5): 054015.

        Eqn. (2)

        :param d
        the distance with the unit of m from the x-ray source

        :return: unit in Gy/s
        """
        k = 17
        a = 2.65 if a is None else a
        return k*Ek_MeV**a* I_avg_A *d**-2
common_used_units = CommonUsedUnit()
if __name__ == '__main__':
    from _logging import logger
    logger.info(DoseRateCalculator().n_interp(10))
    logger.info(DoseRateCalculator().dose_rate_easy(10, 10e-6 * 100 * 100e-3 /1e-6))
    logger.info(
        # 计算结果与论文一致：
        # Under the conditions of an electron-beam energy of 10 MeV and an SAD of 80 cm, beams with a mean current of 1 mA can approximately generate x rays with a mean dose rate of 11.87 Gy/s.
        # [1] Liu F, Zhang L, Shi J, Zha H, Zhu Y, Gao Q, Zhang F, Hu A, Qiu R, Li J, Huang W, Tang C, Chen H, Yan Q, Liu G, Zhang X, He Y, Liu Y, Liu J, Qiu J, Han Y, Wang J, Wang C, Guo C, Men K, Zhu H, Deng X, Wang W, Hu K. MAX-FLASH: A compact multiangle x-ray system for clinical translation of FLASH radiotherapy[J]. Physical Review Applied, 2025, 24(5): 054015.


        DoseRateCalculator().dose_rate_LiuFochengPRApp2025(10,1e-3,80e-2,)
    )
    logger.info(
        # 结果同
        # Nagae D, Yoshida M, Ozaki K, Yamada T, Hasegawa D, Ishiwata J, Shiota Y, Nagase A, Kurihara T, Morishige A, Yamamoto M. Design of a 9-MeV electron linac for high-energy X-ray computed tomography[J]. Nuclear Instruments and Methods in Physics Research Section B: Beam Interactions with Materials and Atoms, 2026, 579: 166233.
        # The X-ray dose rate at a distance of 1 m from the target was 0.1 Gy/min per 100 μA for the 0.95-MeV target and 8.5 Gy/min per 25 μA for the 9-MeV target, confirming that sufficient intensity for CT imaging can be achieved.
        DoseRateCalculator().dose_rate_LiuFochengPRApp2025(9.,25e-6 ,1, )*60)