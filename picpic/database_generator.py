import pandas as pd
from _logging import logger
import grd_parser
import matplotlib.pyplot as plt
import sys
import geom_parser
import numpy as np
import time
from PIL import Image
import os
from scipy import optimize as op



if __name__ == '__main__':
    save_path = r"E:\11-24\database"
    csv_path = r"F:\11-24\optimize1\template.m2d.log.csv"
    csv_name = 'database'
    df = pd.read_csv(csv_path)
    norepeat_df = df.drop_duplicates(subset='m2d_path', keep='first')
    norepeat_df.insert(norepeat_df.shape[1], 'png_path', '')
    norepeat_df.insert(norepeat_df.shape[1], 'Magnetic_field_function_A', '')
    norepeat_df.insert(norepeat_df.shape[1], 'Magnetic_field_function_B', '')
    norepeat_df.insert(norepeat_df.shape[1], 'Magnetic_field_function_C', '')
    norepeat_df.insert(norepeat_df.shape[1], 'Magnetic_field_function_D', '')
    norepeat_df.insert(norepeat_df.shape[1], 'Magnetic_field_function_E', '')
    def f_1(x, A, B, C, D, E):
        return  A * x**4 + B * x**3 + C*x**2+D*x+E

    for i in norepeat_df.index:
        dirStr, ext = os.path.splitext(norepeat_df['m2d_path'][i])
        file = dirStr.split("\\")[-1]
        geom = geom_parser.GEOM(dirStr+'.grd')

        grd = geom.grd
        # 需要拟合的数据组
        title = ' FIELD BZ_ST @LINE_PARTICLE_MOVING$ #1.1'
        x_group = grd.ranges[title][1]['data'][0]
        y_group = grd.ranges[title][1]['data'][1]
        A, B, C, D, E = op.curve_fit(f_1, x_group, y_group)[0]

        plt.figure()
        geom.plot(plt.gca())
        ax = plt.gca()
        ax.set_aspect(1)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            plt.savefig(save_path + "\\" + file + '.png', bbox_inches='tight',dpi=1000)
            norepeat_df.at[i, 'png_path'] = save_path + '\\' + file + '.png'
            logger.info(file + '.png图片保存成功，保存在' + save_path + "\n")
            plt.close()
            norepeat_df.at[i, 'Magnetic_field_function_A'] = A
            norepeat_df.at[i, 'Magnetic_field_function_B'] = B
            norepeat_df.at[i, 'Magnetic_field_function_C'] = C
            norepeat_df.at[i, 'Magnetic_field_function_D'] = D
            norepeat_df.at[i, 'Magnetic_field_function_E'] = E
            logger.info('磁场曲线保存成功:'+'%f * x**4 + %f * x**3 + %f * x**2 + %f * x + %f'%(A,B,C,D,E))
        except:
            logger.info('保存失败')
    norepeat_df.to_csv(save_path + '\\' + csv_name)




    # filename = r"E:\11-24\2\template_20231127_072536_54.grd"
    # dirStr, ext = os.path.splitext(filename)
    # file = dirStr.split("\\")[-1]
    # geom = geom_parser.GEOM(filename)
    # plt.figure()
    # geom.plot(plt.gca())
    # ax = plt.gca()
    # ax.set_aspect(1)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.xticks([])
    # plt.yticks([])
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig(save_path+"\\"+file+'.png',bbox_inches='tight')

