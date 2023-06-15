# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 15:40
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : manualtask.py
# @Software: PyCharm

from simulation.task_manager.task import MagicTemplate, TaskBase

from threading import Lock
class SingleObjectiveOptimizerBase(TaskBase):
    def __init__(self, template_name, working_dir=r'D:\MagicFiles\HPM\12.5GHz\优化',lock=Lock()):
        super(SingleObjectiveOptimizerBase, self).__init__(template_name, working_dir,lock)


    # def get_res(self):
    #     """
    #     获取结果
    #     :return:
    #     """

    def generate_new_param_set(self, *args, **kwargs):
        """
        通过某种优化算法，生成新的参数
        :param args:
        :param kwargs:
        :return:
        """
        return dict()

    def run(self):
        """
        运行整个优化过程
        :return:
        """
        while not self.need_to_finish():
            # self.evaluate()
            params = self.generate_new_param_set()
            self.update(params)


    def need_to_finish(self):
        """
        判断程序是否需要终止
        :return:
        """

    # def evaluate(self, *args, **kwargs):
    #     """
    #     评估结果
    #     :return:
    #     """
    #     print("evaluate of OptimizerBase")





if __name__ == '__main__':
    template = MagicTemplate(r'F:\changeworld\HPMCalc\simulation\template\RBWO-template.m2d',
                             r'D:\MagicFiles\HPM\12.5GHz\优化')
    vars = template.get_variables()

