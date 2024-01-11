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
import typing
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
    def __init__(self,
                 GPT_bin_dir: str = cfg.get_value(cfg.ItemNames.General_Particle_Tracer_bin_dir),
                 GPT_license: str = cfg.get_value(cfg.ItemNames.General_Particle_Tracer_license)
                 ):
        self.GPT_bin_dir = GPT_bin_dir
        self.GPT_license = GPT_license
        self.set_env()
        # self.working_dir = working_dir

    def set_env(self):
        if self.GPT_bin_dir not in os.environ['PATH']:
            os.environ['PATH'] += (self.GPT_bin_dir) + r'\;'
        os.environ['GPTLICENSE'] = self.GPT_license

    def cmd_one_line_command_to_set_env(self):
        return "set path=%path%;" + r"%s\; && set GPTLICENSE=%s " % (self.GPT_bin_dir, self.GPT_license)

    @staticmethod
    def bat_to_one_line_command(bat_filename: str):
        with open(bat_filename, "r") as f:
            s = f.read()
        snew = re.sub(r'\n+', " && ", s)
        return snew

    def cmd_one_line_command_cd_to_dir(self, dir):
        """
        Windows cmd环境下切换到某一工作目录的命令
        :param dir:
        :return:
        """
        return ("cd /d %s" % dir)

    def run_bat(self, batfilepath, workingdir: typing.Union[str, None] = None):
        """
        :param batfilepath: 要求：文件内容中不包含注释、空行
        :param workingdir: 工作目录
        :return:
        """
        batfilepath = os.path.abspath(batfilepath)
        if not workingdir:
            workingdir = os.path.split(batfilepath)[0]
        else:
            workingdir = os.path.abspath(workingdir)
        if os.path.exists(workingdir):
            # cmd = self.command_to_set_env() + ("&& cd %s && %s && " % (workingdir,workingdir.split(':')[0]+':')) + self.bat_to_one_line(batfilepath)
            cmd = self.make_temporary_bat(workingdir, batfilepath)
            self.run_one_line_cmd_command(cmd)

    def run(self, GPT_input_file_path: str, *args, **kwargs):
        self.run_one_line_cmd_command(GPT_input_file_path)

    def run_one_line_cmd_command(self, one_line_cmd_command: str):
        logger.info("Run command: %s" % one_line_cmd_command)
        os.system(one_line_cmd_command)

    def make_temporary_bat(self, working_dir: str, main_bat_path: str):
        """
        新建的bat内容如下：

        cd /d working_dir
        <commands in main_bat_path>

        :param working_dir:
        :param main_bat_path:
        :return:
        """
        cmds = self.cmd_one_line_command_cd_to_dir(working_dir) + "\n"
        with open(main_bat_path, 'r') as f:
            cmds += f.read()
        temp_bat_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'temp.bat')
        with open(temp_bat_path, 'w') as f:
            f.write(cmds)
            logger.debug("Run %s\n内容如下\n%s" % (temp_bat_path, cmds))
        return temp_bat_path

    def delete_result(self, inputfile):
        """
        # TODO
        :param inputfile:
        :return:
        """


default_gptsim = GeneralParticleTracerSim()


def csv_to_gdf(csv_name: str, ):
    filename_noext = os.path.splitext(csv_name)[0]
    gdf_name = '%s.gdf' % filename_noext
    cmd = 'asci2gdf -o %s %s' % (gdf_name, csv_name)
    logger.debug(cmd)
    default_gptsim.run_one_line_cmd_command(cmd)
    return gdf_name


def df_to_gdf(df, gdf_name: str, delete_temp_file=True):
    csv_name = os.path.splitext(gdf_name)[0] + '.txt'
    df.to_csv(csv_name, index=False, sep='\t')
    csv_to_gdf(csv_name)
    if delete_temp_file:
        os.remove(csv_name)


class InputFileTemplateBase(ABC):
    """
    各类模拟软件所需的输入文件模板
    """

    class FileGenerationRecord:
        PATTERN = '\{.+\} => .+'
        def __init__(self, replace_rules: dict, file_path: str):
            self.replace_rules = replace_rules
            self.file_path = file_path

        def __str__(self):
            return '%s => %s\n' % (self.replace_rules, self.file_path)

        @staticmethod
        def from_str(text: str):
            _reprules, filepath = text.split(' => ')
            import json
            reprules = json.loads(_reprules.replace("'", '"'), )
            return InputFileTemplateBase.FileGenerationRecord(reprules, filepath)


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


if __name__ == '__main__':
    test_log = r"""2024-01-10 16:00:12,405 - File "D:\hpmcalc-master\simulation\task_manager\task.py", line 253 - INFO: 当前参数：{'%Current%': 9000, 'comment': ''} => F:\TTO_40GHz\optimize\0110\TT0_40GHz_template4_20240110_160012_30.m2d"""
    text = r"""{'%Current%': 9000, 'comment': ''} => F:\TTO_40GHz\optimize\0110\TT0_40GHz_template4_20240110_160012_30.m2d"""
    record = InputFileTemplateBase.FileGenerationRecord.from_str(text)
    str(record)
