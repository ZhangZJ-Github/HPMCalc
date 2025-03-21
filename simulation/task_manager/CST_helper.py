# -*- coding: utf-8 -*-
# @Time    : 2024/10/27 15:22
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : CST_helper.py
# @Software: PyCharm
import pandas

from scipy.interpolate import interp1d
def param_df_to_param_text(df:pandas.DataFrame)->str:
    """
将形如
   GDT_z  GDT_y    dz1  Hole1Dz  Hole2Dz  SwitchCavDz  SwitchCavDy
0  275.1     20  18.48     23.9     19.1         31.9           26
的df转化为CST ”load parameter from file“所支持的文本

    :return:
    """
    # df = pandas.read_csv(r"H:\Users\Zhang\Desktop/1.csv")
    s = ''
    for col in df.columns:
        s += '%s="%s" \n' % (col, df[col][0])
    # print(s)
    return s


def get_interpolator(E_data):
    return interp1d(E_data[:, 0].real, E_data[:, 1])


if __name__ == '__main__':
    print(param_df_to_param_text(pandas.read_csv(r"H:\Users\Zhang\Desktop/1.csv")))