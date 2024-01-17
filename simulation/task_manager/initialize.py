# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 14:05
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : initialize.py
# @Software: PyCharm
import os
import typing

import pandas

import simulation.task_manager._base


class Initializer:
    def __init__(self, filename, bound_factor=0.1):
        """
        :param filename:
        :param bound_factor: between 0 and 1. 当未知名上下界时，默认将其设为最后一组初始值基础上下浮动bound_factor
        """
        # TODO: 处理表中无实际内容的情况
        if not os.path.exists(filename):
            self.make_new_initial_csv(filename, )
            raise RuntimeError('用于初始化的"%s"不存在，请手动填写其内容，或指定其他初始化文件' % (filename))
        self.filename = filename
        initial_df = pandas.read_csv(filename, encoding=simulation.task_manager._base.CSV_ENCODING)
        self.initial_df = initial_df[initial_df.columns[1:]]  # 去除备注列
        self.N_initial = len(self.initial_df) - 5
        self.init_params: typing.List[typing.Dict[str, float]] = [
            {col: self.initial_df[col][i] for col in self.initial_df} for i in range(self.N_initial)]
        self.init_params_df: pandas.DataFrame = self.initial_df.iloc[:self.N_initial]  # 初始参数
        self.index_to_param_name = lambda i: self.initial_df.columns[i]
        self.param_name_to_index = {
            self.initial_df.columns[i]: i for i in range(len(self.initial_df.columns))
        }
        # bound_factor = 0.1
        self.bound_factor = bound_factor
        default_bound_df = pandas.concat(
            [self.init_params_df.iloc[-1] * fac for fac in (1 - self.bound_factor, 1 + self.bound_factor)], axis=1, ).T
        self.lower_bound = self.initial_df.iloc[self.N_initial + 0].fillna(default_bound_df.min(axis=0)).values  # 参数下边界
        self.upper_bound = self.initial_df.iloc[self.N_initial + 1].fillna(default_bound_df.max(axis=0)).values  # 上边界
        self.precision_df = self.initial_df.iloc[self.N_initial + 2]  # .fillna(0) # 为NaN表示不做截断
        self.precision = self.precision_df.values  # shape (N_params,)

    def update(self):
        self.__init__(self.filename)

    @staticmethod
    def make_new_initial_csv(filename, variables: typing.List[str] = None):
        if variables is None: variables = []
        _df = pandas.DataFrame(columns=list(variables))
        row_names = """变量名
初始值
下界
上界
精度
计算偏微分的步长
单位步长""".split('\n')
        _df = pandas.concat([pandas.DataFrame({row_names[0]: row_names[1:]}), _df])
        _df.to_csv(filename, index=False, encoding=simulation.task_manager._base.CSV_ENCODING)
        os.system("start %s" % filename)


if __name__ == '__main__':
    initializer = Initializer.make_new_initial_csv('initial.temp.csv')
