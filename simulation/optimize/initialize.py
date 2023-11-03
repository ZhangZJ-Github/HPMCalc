# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 14:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : initialize.py
# @Software: PyCharm
import os
import typing

import pandas

import simulation.task_manager.task


class Initializer:
    def __init__(self, filename):
        self.filename = filename
        initial_df = pandas.read_csv(filename, encoding=simulation.task_manager.task.CSV_ENCODING)
        self.initial_df = initial_df[initial_df.columns[1:]]  # 去除备注列
        self.N_initial = len(self.initial_df) - 5
        self.init_params: typing.List[typing.Dict[str, float]] = [
            {col: self.initial_df[col][i] for col in self.initial_df} for i in range(self.N_initial)]
        self.index_to_param_name = lambda i: self.initial_df.columns[i]
        self.param_name_to_index = {
            self.initial_df.columns[i]: i for i in range(len(self.initial_df.columns))
        }
        self.lower_bound = self.initial_df.iloc[self.N_initial + 0].values  # 参数下边界
        self.upper_bound = self.initial_df.iloc[self.N_initial + 1].values  # 上边界
        self.precision_df = self.initial_df.iloc[self.N_initial + 2]
        self.precision = self.precision_df.values  # shape (N_params,)
    def update(self):
        self.__init__(self.filename)

    @staticmethod
    def make_new_initial_csv(filename, simtask: simulation.task_manager.task.TaskBase):
        _df = pandas.DataFrame(columns=list(simtask.template.get_variables()))
        row_names = """备注
初始值
下界
上界
精度
计算偏微分的步长
单位步长""".split('\n')
        _df = pandas.concat([pandas.DataFrame({row_names[0]: row_names[1:]}), _df])
        _df.to_csv(filename, index=False, encoding=simulation.task_manager.task.CSV_ENCODING)
        os.system("start %s" % filename)
