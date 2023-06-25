# -*- coding: utf-8 -*-
# @Time    : 2023/6/25 21:07
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test.py
# @Software: PyCharm
import optimize_HPM
optimize_HPM.HPMSim(r"F:\changeworld\HPMCalc\simulation\template\RSSSE\RSSE_template.m2d",
                  r'D:\MagicFiles\HPM\12.5GHz\优化4', 11.7e9, 1e9).clean_folder(2e-3)
