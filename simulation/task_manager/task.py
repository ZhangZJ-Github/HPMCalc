# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 14:02
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : task.py
# @Software: PyCharm

import datetime
import os
import os.path
import os.path
import re
import shutil
import time
from abc import abstractmethod
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas
import sw_to_MAGIC_commands
from _logging import logger
from threading import Lock

# import simulation.task_manager.manual_task
CSV_ENCODING = 'gbk'


class MagicTemplate:
    VARIABLE_PATTERN = r"%[\w.]+%"

    def __init__(self, filename, working_dir: str,lock=Lock()):
        self.filename = filename
        self.working_dir = working_dir
        self.output_prefix = os.path.join(self.working_dir,
                                          os.path.split(os.path.splitext(self.filename)[0])[1])
        self.lock = lock

        with open(filename, 'r') as f:
            self.text = f.read()

    def __str__(self):
        return "template: %s\nworking dir: %s" % (self.filename, self.working_dir)

    def get_variables(self):
        return list(OrderedDict.fromkeys(re.findall(self.VARIABLE_PATTERN, self.text)))

    def generate(self, replace_rules: dict):
        """
        :param replace_rules: {old_str: new_str,}
        :return:
        """

        template_text = self.text
        for old in replace_rules:
            template_text = template_text.replace(old, replace_rules[old])
        return template_text

    def new_m2d_file_name(self):
        self.lock.acquire()

        timestamp_ns = time.time_ns()
        time.sleep(0.1)
        self.lock.release()
        return self.output_prefix + datetime.datetime.fromtimestamp(timestamp_ns/1e9).strftime(
            "_%Y%m%d_%H%M%S_") + ("%09d.m2d" % (timestamp_ns % 1e9))[1:]

    def to_m2d(self, replace_rules: dict):
        file_path = self.new_m2d_file_name()
        with open(file_path, 'w') as f:
            f.write(self.generate(replace_rules))
        return file_path


from threading import Lock


class TaskBase:
    MAGIC_SOLVER_PATH = r"G:\Program Files\Magic Tools\magic2d_Sng.exe"
    colname_score = "score"
    colname_path = "m2d_path"
    colname_timestamp = 'timestamp'

    def __init__(self, template_name, working_dir=r'D:\MagicFiles\HPM\12.5GHz\优化', lock: Lock = Lock()):
        self.template = MagicTemplate(template_name, working_dir)
        self.log_file_name = os.path.join(working_dir, os.path.split(template_name)[1] + ".log.csv")
        # if not os.path.exists(self.log_file_name):
        # self.log_df = pandas.DataFrame(#columns=list(self.template.get_variables())
        #                                # + [self.colname_path,self.colname_score]
        #                                )

        self.lock = lock
        # self.lock.acquire()
        # self.load_log_csv()
        # self.lock.release()

        # else:
        #     self.log_df = pandas.read_csv(self.log_file_name,encoding='gbk')
        logger.info("使用模板：%s" % self.template)

    def save_log_csv(self):
        if os.path.exists(self.log_file_name):
            df = pandas.concat([self.load_log(), self.log_df], axis=0)
        else:
            df = self.log_df
        df.to_csv(self.log_file_name, encoding=CSV_ENCODING, index=False)

    # def load_log_csv(self):
    #     """
    #     从硬盘中加载log.csv，并与当前df比较
    #     :return:
    #     """
    #     if os.path.exists(self.log_file_name):
    #         self.log_df = pandas.merge(pandas.read_csv(self.log_file_name, encoding=CSV_ENCODING), self.log_df,
    #                                    on=None, how='outer')#.drop_duplicates()

    def log(self, params: dict, m2d_path: str):
        """
        将修改的参数和主要结果记录到文件中
        :return:
        """
        newdata = params.copy()
        res = self.get_res(m2d_path)
        res[self.colname_timestamp] = time.time()
        res[self.colname_score] = self.evaluate(res)
        res[self.colname_path] = m2d_path

        newdata.update(res)
        # self.load_log_csv()
        self.log_df = pandas.DataFrame.from_dict({key: [newdata[key]] for key in newdata})
        # self.log_df.loc[len(self.log_df)] = numpy.nan
        # for key in newdata:
        #     if key not in self.log_df.columns:
        #         self.log_df[key] = numpy.nan
        #     self.log_df[key][len(self.log_df) - 1] = newdata[key]
        self.lock.acquire()

        self.save_log_csv()
        self.lock.release()
        return self.log_df

    def load_log(self) -> pandas.DataFrame:
        return pandas.read_csv(self.log_file_name, encoding=CSV_ENCODING)

    def evaluate(self, res: dict):
        pass

    def params_check(self, params: dict) -> bool:
        """
        检查输入参数是否合法
        :param params:
        :return:
        """
        return True

    def update(self, param_set: dict):
        """
        按照给定的参数运行模拟，自动获取结果，更新log
        :param param_set:
        :return: 评分
        """
        params_check_status = self.params_check(param_set)
        if not params_check_status:
            logger.info("无效参数：%s" % (param_set))
            return 0.
        logger.info("当前参数：%s" % param_set)
        m2d_path = self.template.to_m2d({key: str(param_set[key]) for key in param_set})
        cmd = '"%s" %s' % (self.MAGIC_SOLVER_PATH, m2d_path)
        logger.info("Run command:\n%s" % cmd)
        os.system(cmd)

        log_df = self.log(param_set, m2d_path)
        last_logged_data = log_df.iloc[len(log_df) - 1]
        logger.info(str(last_logged_data))

        return last_logged_data[self.colname_score]

    @abstractmethod
    def get_res(self, m2d_path: str) -> dict:
        """
        获取用户关心的数据
        :param m2d_path:
        :return:
        """
        return dict()


class ManualTask:
    MAGIC_SOLVER_PATH = TaskBase.MAGIC_SOLVER_PATH

    def __init__(self, children_sldprt_name, m2d_template_name, replace_marker: str,
                 folder_and_prefix, parent_sldprt_name, description):
        self.parent_sldprt_name = parent_sldprt_name
        self.children_sldprt_name = children_sldprt_name
        self.m2d_template_name = m2d_template_name
        self.folder, self.prefix = os.path.split(folder_and_prefix)

        self.description = description
        self.replace_marker = replace_marker

    def run(self, ax=plt.gca()):
        os.makedirs(self.folder, exist_ok=True)
        with open(self.m2d_template_name, 'r') as f:
            txt = f.read()
        new_txt = txt.replace(self.replace_marker,
                              sw_to_MAGIC_commands.sldprt_to_MAGIC_commands(self.children_sldprt_name, ax))
        m2d_file = os.path.join(self.folder, (self.prefix + '.m2d'))
        with open(m2d_file, 'w') as f:
            f.write(new_txt)
            logger.info("%s written." % m2d_file)
        with open(os.path.join(self.folder, self.prefix + '.about.txt'), 'w') as f:
            f.write(self.description)

        shutil.copy(self.parent_sldprt_name,
                    os.path.join(self.folder, (self.prefix + os.path.splitext(self.parent_sldprt_name)[1]))
                    )
        command = 'start "%s" %s' % (self.MAGIC_SOLVER_PATH, m2d_file)
        os.system(command)
        logger.info(command)
        logger.info(self.folder)
        logger.info(self.description)


if __name__ == '__main__':
    plt.ion()
    tsk = ManualTask([
        r"D:\MagicFiles\HPM\12.5GHz\swCATHODE.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swANODE.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swEMITTER.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swPORT_LEFT.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swPORT_RIGHT.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swProbeExtractionCav.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swProbePremodulationCav.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swProbeReflCav.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swProbeReflCav2.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swProbeSWS.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swRefl.sldprt",
        r"D:\MagicFiles\HPM\12.5GHz\swMAIN.sldprt",

        # r"D:\MagicFiles\HPM\12.5GHz\swOutBarrier.sldprt",

    ],
        r'D:\MagicFiles\HPM\12.5GHz\RBWO-template.m2d',
        '!!! Modelling commands here !!!',
        input(r"输入文件路径前缀，如'D:\MagicFiles\HPM\12.5GHz\自动化\test01\改周期长度2\sim.toc'："),
        r"D:\MagicFiles\HPM\12.5GHz\HPM-全金属外壳.SLDPRT",
        input("输入描述信息，可省略：\n"))
    tsk.run()
    plt.legend()
    plt.show()
