# -*- coding: utf-8 -*-
# @Time    : 2024/1/9 15:50
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : simulator.py
# @Software: PyCharm
"""
用于管理各种模拟与Python的接口
"""

import datetime
import os
import os.path
import os.path
import re
import shutil
import time
from collections import OrderedDict

import matplotlib

matplotlib.use('tkagg')
from _logging import logger
from simulation.conf.config import Config

# import simulation.task_manager.manual_task
CSV_ENCODING = 'gbk'
cfg = Config.read_json_file()
from threading import Lock
from abc import ABC, abstractmethod
from filenametool import ExtTool


class SimulationExecutor(ABC):
    @abstractmethod
    def run(self, inputfile, *args, **kwargs):
        logger.warning("注意：实际上没有执行任何模拟！")

    @abstractmethod
    def delete_result(self, inputfile):
        """
        删除和inputfile相关联的结果文件
        :param inputfile:
        :return:
        """


class MAGICSim(SimulationExecutor):
    EXECUTOR_PATH = cfg.get_value(Config.ItemNames.Magic_executable_path)

    def __init__(self):
        if not os.path.exists(self.EXECUTOR_PATH):
            raise RuntimeError("请指定正确的MAGIC求解器路径！")

    def run(self, inputfile, *args, **kwargs):
        cmd = '"%s" %s' % (self.EXECUTOR_PATH, inputfile)
        logger.info("Run command:\n%s" % cmd)
        os.system(cmd)

    def delete_result(self, inputfile):
        big_files_to_delete = []
        et = ExtTool(os.path.splitext(inputfile)[0])
        for filetype in [
            ExtTool.FileType.grd, ExtTool.FileType.par,
            ExtTool.FileType.fld, ExtTool.FileType.toc
        ]:
            filename = et.get_name_with_ext(filetype)
            if os.path.exists(filename):
                os.remove(filename)
            big_files_to_delete.append(filename)
        logger.info("已经删除的文件：\n%s" % big_files_to_delete)


class GeneralParticleTracerSim(SimulationExecutor):
    # TODO
    pass


class InputFileTemplateBase(ABC):
    """
    各类模拟软件所需的输入文件模板
    """

    def __init__(self, filename, working_dir: str, lock=Lock(), variable_pattern=r"%[\w.]+%",
                 encoding='utf-8'  # or 'gbk'
                 ):
        """
        :param filename: 模板文件的路径
        :param working_dir: 利用模板生成的文件所保存的文件夹
        :param variable_pattern: 模板中待替换的字符串的形式（用正则表达式表示）
        :param lock:
        """
        self.filename = filename
        self.working_dir = working_dir
        self.output_prefix = os.path.join(
            self.working_dir,
            os.path.split(os.path.splitext(self.filename)[0])[1])
        self.lock = lock
        self.variable_pattern = variable_pattern
        self.encoding = encoding

        with open(filename, 'r', encoding=self.encoding) as f:
            self.text = f.read()

    def copy_template_to_working_dir(self):
        shutil.copy(
            self.filename,
            os.path.join(self.working_dir,
                         os.path.split(self.filename)[1]))

    def __str__(self):
        return "template: %s\nworking dir: %s" % (self.filename,
                                                  self.working_dir)

    def get_variables(self):
        return list(
            OrderedDict.fromkeys(re.findall(self.variable_pattern, self.text)))

    def generate(self, replace_rules: dict):
        """
        只生成文件的内容，但不写入硬盘
        :param replace_rules: {old_str: new_str,}
        :return:
        """

        template_text = self.text
        for old in replace_rules:
            template_text = template_text.replace(old, replace_rules[old])
        return template_text

    def new_file_name(self, *arg, **kwargs):
        """
        :param *arg:
        :param **kwargs:
        :return: 根据模板生成的文件的文件名
        """
        self.lock.acquire()

        timestamp_ns = time.time_ns()
        time.sleep(0.1)
        self.lock.release()
        return self.output_prefix + datetime.datetime.fromtimestamp(
            timestamp_ns // 1e9).strftime("_%Y%m%d_%H%M%S_") + ("%02d.m2d" % (
                (timestamp_ns % 1e9) // 1e7))

    def generate_and_to_disk(self, replace_rules: dict):
        file_path = self.new_file_name()
        with open(file_path, 'w', encoding=self.encoding) as f:
            f.write(self.generate(replace_rules))
        return file_path


class MagicTemplate(InputFileTemplateBase):
    VARIABLE_PATTERN = r"%[\w.]+%"
    M2D_ENCODING = 'utf-8'  # or 'gbk'

    def __init__(self, filename, working_dir: str, lock=Lock()
                 ):
        super(MagicTemplate, self).__init__(
            filename, working_dir, lock=lock,
            variable_pattern=self.VARIABLE_PATTERN,
            encoding=self.M2D_ENCODING
        )
