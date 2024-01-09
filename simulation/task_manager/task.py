# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 14:02
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : task.py
# @Software: PyCharm

import os.path
import os.path

import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas
from deprecated.sphinx import deprecated
from simulation.task_manager.simulator import *

# import simulation.task_manager.manual_task
CSV_ENCODING = 'gbk'
cfg = Config.read_json_file()


class TaskBase(ABC):
    class Colname:
        score = "score"
        path = "m2d_path"
        timestamp = 'timestamp'
        comment = 'comment'

    def __init__(self,
                 template: InputFileTemplateBase,
                 simulation_executor: SimulationExecutor = MAGICSim(),
                 lock: Lock = Lock(),
                 ):
        self.template = template
        self.log_file_name = os.path.join(
            template.working_dir,
            os.path.split(template.filename)[1] + ".log.csv")
        self.simulation_executor = simulation_executor

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

    def find_old_res(self, params: dict, precisions: dict = {}) -> str:
        # TODO: 检查有效性
        if os.path.exists(self.log_file_name):
            self.lock.acquire()
            # logger.info('self.lock.acquired')
            df = self.load_log()
            self.lock.release()
            # logger.info('self.lock.released')

            numeric_params = {}
            for key in params:
                if key.startswith('%'):
                    numeric_params[key] = params[key]

            df_this_numeric_params = pandas.DataFrame(numeric_params, index=df.index)
            df_precision = pandas.DataFrame({key: precisions.get(key, 0) for key in df_this_numeric_params.columns},
                                            index=df.index)
            mask = (((df[df_this_numeric_params.columns] - df_this_numeric_params).abs() - df_precision) <= 0).all(
                axis=1)
            m2d_paths = df[self.Colname.path][mask]
            if m2d_paths.any():
                logger.info("找到了之前的记录！\n%s => %s" % (params, m2d_paths.values[0]))
                return m2d_paths.values[0]
            return ''

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
        logger.info("old_m2d_path=%s" % old_m2d_path)

        if old_m2d_path:
            return self.log_and_info(param_set, old_m2d_path)

        m2d_path = self.template.generate_and_to_disk({key: str(param_set[key])
                                                       for key in param_set})
        self.last_generated_m2d_path = m2d_path
        logger.info("当前参数：%s => %s" % (param_set, m2d_path))
        self.simulation_executor.run(m2d_path)
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
        # 一般不会并行执行，因此不加锁
        log_df = self.load_log()
        for i in range(len(log_df)):
            path = log_df[self.Colname.path][i]

            res = self.get_res(path)
            score = self.evaluate(res)
            # log_df[self.colname_score] [i]= score
            res[self.Colname.score] = score
            for key in res:
                if key not in log_df.columns:
                    log_df[key] = pandas.NA
                log_df[key][i] = res[key]

        self.rewrite_log_csv(log_df)


    def clean_working_dir(self, score_threshold):
        """
        删除文件夹中不必要的（低分）结果
        :return:
        """
        log_df = self.load_log()
        # index_to_delete = [log_df[self.colname_score] < score_threshold ]
        m2d_paths_to_delete = log_df[self.Colname.path][
            log_df[self.Colname.score] < score_threshold]  # .tolist()

        for m2d_path in m2d_paths_to_delete:
            self.simulation_executor.delete_result(m2d_path)


class MAGICTaskBase(TaskBase):
    MAGIC_SOLVER_PATH = MAGICSim.EXECUTOR_PATH

    def __init__(self,
                 template_name,
                 working_dir=r"E:\ref_test",
                 lock: Lock = Lock(),
                 simulation_executor=MAGICSim()):
        super(MAGICTaskBase, self).__init__(MagicTemplate(template_name, working_dir, lock), simulation_executor, lock)
        self.last_generated_m2d_path = ''  # 最近一次生成的m2d文件路径


class ManualTask:
    MAGIC_SOLVER_PATH = MAGICTaskBase.MAGIC_SOLVER_PATH

    @deprecated(version="since20240109", reason="暂无将solidworks模型导入MAGIC的需求，因此不再维护此功能")
    def __init__(self, children_sldprt_name, m2d_template_name,
                 replace_marker: str, folder_and_prefix, parent_sldprt_name,
                 description):
        self.parent_sldprt_name = parent_sldprt_name
        self.children_sldprt_name = children_sldprt_name
        self.m2d_template_name = m2d_template_name
        self.folder, self.prefix = os.path.split(folder_and_prefix)

        self.description = description
        self.replace_marker = replace_marker

    @deprecated(version="since20240109", reason="暂无将solidworks模型导入MAGIC的需求，因此不再维护此功能")
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
