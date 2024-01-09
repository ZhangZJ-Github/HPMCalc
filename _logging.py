# -*- coding: utf-8 -*-
# @Time    : 2022/7/31 17:36
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : logger.py
# @Software: PyCharm
import logging

logger_name = "zzj_log"
fmt = logging.Formatter('%(asctime)s - File "%(pathname)s", line %(lineno)d - %(levelname)s: %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(fmt)

logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # 用于输出至文件
    file_log_handler = logging.FileHandler("log.txt", encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    file_log_handler.setFormatter(fmt)
    # logger绑定处理对象
    logger.addHandler(file_log_handler)
    logger.addHandler(stream_handler)


# def get_logger():
#     return logger
