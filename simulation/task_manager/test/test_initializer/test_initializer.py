# -*- coding: utf-8 -*-
# @Time    : 2024/1/18 22:36
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : test_initializer.py
# @Software: PyCharm

from simulation.task_manager.initialize import Initializer
from simulation.task_manager.simulator import MagicTemplate

initializer = Initializer.make_new_initial_csv(
    'initial.csv',
    MagicTemplate(r'template.m2d', '.').get_variables())
