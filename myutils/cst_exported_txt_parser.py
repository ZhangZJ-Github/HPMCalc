# -*- coding: utf-8 -*-
# @Time    : 2023/1/8 19:17
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : cst_exported_txt_parser.py
# 用于解析CST导出的txt文件
# @Software: PyCharm

import io
import os.path
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy
import pandas
from scipy import signal
from scipy.fftpack import fft
# plt.ion()

DEFAULT_OUT_DIR = os.path.join(os.path.split(__file__)[0], ".out")
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)


def parse(path: str, column_names=["t", "signal"]):
    with open(path, 'r') as f:
        lines = f.readlines()
    data_indexes = [[]]
    label_indexes = []
    last_line_is_label = False
    i = 0
    for line in lines:
        if line.startswith("#"):
            if not last_line_is_label:
                data_indexes[-1].append(i)
                label_indexes.append([i])
            last_line_is_label = True
        else:
            if last_line_is_label:
                label_indexes[-1].append(i)
                data_indexes.append([i])
            last_line_is_label = False
        i += 1
    data_indexes = data_indexes[1:]
    data_indexes[-1].append(-1)
    labels = []
    datas = []
    dfs = []

    for j in range(len(data_indexes)):
        label_index = label_indexes[j]
        data_index = data_indexes[j]
        labels.append(lines[label_index[0]:label_index[1]])
        data_str = "".join(lines[data_index[0]:data_index[1]])
        datas.append(data_str)
        df = pandas.read_csv(io.StringIO(data_str), sep="\t", header=None)
        df.columns = column_names
        dfs.append(df)
    return dfs, labels

