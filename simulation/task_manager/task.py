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
from collections import OrderedDict
from threading import Lock
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas
from _logging import logger

# import simulation.task_manager.manual_task
CSV_ENCODING = 'gbk'


class MagicTemplate:
    VARIABLE_PATTERN = r"%[\w.]+%"
    M2D_ENCODING = 'utf-8'  # or 'gbk'

    def __init__(self, filename, working_dir: str, lock=Lock()):
        self.filename = filename
        self.working_dir = working_dir
        self.output_prefix = os.path.join(
            self.working_dir,
            os.path.split(os.path.splitext(self.filename)[0])[1])
        self.lock = lock

        with open(filename, 'r', encoding=MagicTemplate.M2D_ENCODING) as f:
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
            OrderedDict.fromkeys(re.findall(self.VARIABLE_PATTERN, self.text)))

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
        return self.output_prefix + datetime.datetime.fromtimestamp(
            timestamp_ns // 1e9).strftime("_%Y%m%d_%H%M%S_") + ("%02d.m2d" % (
                (timestamp_ns % 1e9) // 1e7))

    def to_m2d(self, replace_rules: dict):
        file_path = self.new_m2d_file_name()
        with open(file_path, 'w', encoding=MagicTemplate.M2D_ENCODING) as f:
            f.write(self.generate(replace_rules))
        return file_path


from threading import Lock
from abc import ABC, abstractmethod


class TaskBase(ABC):
    MAGIC_SOLVER_PATH = r"G:\Program Files\Magic Tools\magic2d_Sng.exe"

    class Colname:
        score = "score"
        path = "m2d_path"
        timestamp = 'timestamp'
        comment = 'comment'

    def __init__(self,
                 template_name,
                 working_dir=r'D:\MagicFiles\HPM\12.5GHz\优化',
                 lock: Lock = Lock()):
        self.template = MagicTemplate(template_name, working_dir)
        self.log_file_name = os.path.join(
            working_dir,
            os.path.split(template_name)[1] + ".log.csv")
        self.last_generated_m2d_path = ''  # 最近一次生成的m2d文件路径
        if not os.path.exists(self.MAGIC_SOLVER_PATH):
            raise RuntimeError("请指定正确的MAGIC求解器路径！")

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
        logger.info("====为%s加锁====" % self.log_file_name)
        self.lock.acquire()
        header = False if os.path.exists(self.log_file_name) else True
        self.log_df.to_csv(self.log_file_name,
                           encoding=CSV_ENCODING,
                           index=False,
                           mode='a',
                           header=header)
        self.lock.release()
        logger.info("====%s已解锁====" % self.log_file_name)

    def rewrite_log_csv(self, df):
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
        res[self.Colname.timestamp] = time.time()
        res[self.Colname.score] = self.evaluate(res)
        res[self.Colname.path] = m2d_path

        newdata.update(res)
        # self.load_log_csv()
        self.log_df = pandas.DataFrame.from_dict(
            {key: [newdata[key]]
             for key in newdata})
        # self.log_df.loc[len(self.log_df)] = numpy.nan
        # for key in newdata:
        #     if key not in self.log_df.columns:
        #         self.log_df[key] = numpy.nan
        #     self.log_df[key][len(self.log_df) - 1] = newdata[key]
        self.save_log_csv()
        return self.log_df

    def load_log(self) -> pandas.DataFrame:
        return pandas.read_csv(self.log_file_name, encoding=CSV_ENCODING)

    @abstractmethod
    def evaluate(self, res: dict):
        pass

    @abstractmethod
    def params_check(self, params: dict) -> bool:
        """
        检查输入参数是否合法，若不合法，直接修改传入的字典。
        log.csv中记录的是此处修改之后的参数值。
        :param params:
        :return: True if 修改成功
        """
        pass

    def find_old_res(self, params: dict) -> str:
        # TODO: 检查有效性
        if os.path.exists(self.log_file_name):
            self.lock.acquire()
            # logger.info('self.lock.acquired')
            df = self.load_log()
            self.lock.release()
            # logger.info('self.lock.released')
            condition = pandas.Series([True] * len(df))
            for key in params:
                if key.startswith('%'):
                    old_condition = condition
                    condition = condition & (df[key] == params[key])
                    logger.info(
                        'params[%s] = %s, condition.any() = %s, condition[condition] = %s' % (
                            key, params[key], condition.any(), condition[condition]))
                    if not condition.any():
                        logger.info(
                            'old_condition[df[key] == params[key]] = %s' % (old_condition[df[key] == params[key]]))
                        # logger.info('df[key][df[key] == params[key]] = \n%s'%(df[key][df[key] == params[key]]))
                        break
            logger.info('condition = %s' % (condition))
            old_res_line = df[condition]
            logger.info('len(old_res_line) = %d' % (len(old_res_line)))
            if len(old_res_line):
                # logger.info("old_res_line = %s" % old_res_line)
                old_m2d_path = old_res_line[self.Colname.path].values[
                    len(old_res_line) - 1]
                logger.info("找到了之前的记录：%s" % old_m2d_path)
                return old_m2d_path
        return ""  # means not found

    def log_and_info(self, param_set, m2d_path):
        try:
            log_df = self.log(param_set, m2d_path)
            last_logged_data = log_df.iloc[len(log_df) - 1]
            logger.info("\n%s" % str(last_logged_data))
            score = last_logged_data[self.Colname.score]
            logger.info('path = %s\nscore = %s' % (last_logged_data[self.Colname.path], score))
            return score
        except (KeyError, FileNotFoundError, TypeError, IndexError,
                pandas.errors.ParserError, PermissionError) as e:
            logger.warning("记录结果时报错，已忽略：\n%s" % e)
            return 0.

    def update(self, param_set: dict, comment: str = ''):
        """
        按照给定的参数运行模拟，自动获取结果，更新log
        :param param_set:
        :return: 评分
        """
        param_set[self.Colname.comment] = comment
        params_check_status = self.params_check(param_set)
        if not params_check_status:
            logger.warning("无效参数：%s" % (param_set))
            return 0.
        logger.info('Finding old result...')
        old_m2d_path = self.find_old_res(param_set)

        if old_m2d_path:
            return self.log_and_info(param_set, old_m2d_path)

        m2d_path = self.template.to_m2d(
            {key: str(param_set[key])
             for key in param_set})
        self.last_generated_m2d_path = m2d_path
        logger.info("当前参数：%s => %s" % (param_set, m2d_path))

        cmd = '"%s" %s' % (self.MAGIC_SOLVER_PATH, m2d_path)
        logger.info("Run command:\n%s" % cmd)
        os.system(cmd)
        return self.log_and_info(param_set, m2d_path)

    @abstractmethod
    def get_res(self, m2d_path: str) -> dict:
        """
        获取用户关心的数据
        :param m2d_path:
        :return:
        """
        return dict()

    def re_evaluate(self):
        """
        对log.csv中的每条记录重新打分
        :return:
        """
        # 一般不并行执行，因此不加锁
        log_df = self.load_log()
        for i in range(len(log_df)):
            path = log_df[self.Colname.path][i]

            res = self.get_res(path)
            score = self.evaluate(res)
            # log_df[self.colname_score] [i]= score
            res[self.Colname.score] = score
            for key in res:
                log_df[key][i] = res[key]

        self.rewrite_log_csv(log_df)

    def clean_folder(self, score_threshold):
        """
        删除文件夹中不必要的（低分）结果
        :return:
        """
        log_df = self.load_log()
        # index_to_delete = [log_df[self.colname_score] < score_threshold ]
        m2d_paths_to_delete = log_df[self.Colname.path][
            log_df[self.Colname.score] < score_threshold]  # .tolist()
        big_files_to_delete = []
        from total_parser import ExtTool
        import os

        for m2d_path in m2d_paths_to_delete:
            et = ExtTool(os.path.splitext(m2d_path)[0])
            for filetype in [
                ExtTool.FileType.grd, ExtTool.FileType.par,
                ExtTool.FileType.fld, ExtTool.FileType.toc
            ]:
                filename = et.get_name_with_ext(filetype)
                if os.path.exists(filename):
                    os.remove(filename)
                big_files_to_delete.append(filename)
        # log_df.info(log_df[self.colname_score][index_to_delete])
        logger.info("已经删除的文件：\n%s" % big_files_to_delete)


class ManualTask:
    MAGIC_SOLVER_PATH = TaskBase.MAGIC_SOLVER_PATH

    def __init__(self, children_sldprt_name, m2d_template_name,
                 replace_marker: str, folder_and_prefix, parent_sldprt_name,
                 description):
        self.parent_sldprt_name = parent_sldprt_name
        self.children_sldprt_name = children_sldprt_name
        self.m2d_template_name = m2d_template_name
        self.folder, self.prefix = os.path.split(folder_and_prefix)

        self.description = description
        self.replace_marker = replace_marker

    def run(self, ax=plt.gca()):
        import sw_to_MAGIC_commands
        os.makedirs(self.folder, exist_ok=True)
        with open(self.m2d_template_name, 'r') as f:
            txt = f.read()
        new_txt = txt.replace(
            self.replace_marker,
            sw_to_MAGIC_commands.sldprt_to_MAGIC_commands(
                self.children_sldprt_name, ax))
        m2d_file = os.path.join(self.folder, (self.prefix + '.m2d'))
        with open(m2d_file, 'w') as f:
            f.write(new_txt)
            logger.info("%s written." % m2d_file)
        with open(os.path.join(self.folder, self.prefix + '.about.txt'),
                  'w') as f:
            f.write(self.description)

        shutil.copy(
            self.parent_sldprt_name,
            os.path.join(
                self.folder,
                (self.prefix + os.path.splitext(self.parent_sldprt_name)[1])))
        command = 'start "%s" %s' % (self.MAGIC_SOLVER_PATH, m2d_file)
        os.system(command)
        logger.info(command)
        logger.info(self.folder)
        logger.info(self.description)


if __name__ == '__main__':
    plt.ion()
    tsk = ManualTask(
        [
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
        input(
            r"输入文件路径前缀，如'D:\MagicFiles\HPM\12.5GHz\自动化\test01\改周期长度2\sim.toc'："
        ),
        r"D:\MagicFiles\HPM\12.5GHz\HPM-全金属外壳.SLDPRT",
        input("输入描述信息，可省略：\n"))
    tsk.run()
    plt.legend()
    plt.show()
