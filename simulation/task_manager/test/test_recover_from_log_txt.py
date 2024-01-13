# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 19:43
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : test_recover_from_log_txt.py
# @Software: PyCharm
from simulation.optimize.hpm.hpm import HPMSim

log_text = ''
with open('test_log.txt', 'r', encoding='utf-8') as f:
    log_text = f.read()

mytask = HPMSim(__file__, '.', )
mytask.recorver_log_df_from_log_text(log_text)
