# -*- coding: utf-8 -*-
# @Time    : 2024/10/21 21:16
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : vacuum_breakdown.py
# @Software: PyCharm
import matplotlib
import numpy
import shapely

matplotlib.use('tkagg')
import matplotlib .pyplot as plt


def Kilpatrick(E_unit_in_MV_m):
    """
    Ref: 2023 小型化微波超构材料速调管的研究__张宣铭__电子科技大学

    真空中随频率变化的最大微波击穿场强通常采用半经验公式 Kilpatrick 准则来预测

    实际工程允许的表面电场强度通常取据此估计值的1.4倍
    
    该公式也见诸
    
    Thomas P. Wangler. RF Linear Accelerators[M]. 1st ed. John Wiley & Sons, Ltd, 2008.
    Eqn. (5.80)

    Dolgashev V A. Design Criteria for High-Gradient Radio-Frequency Linacs[J]. Applied Sciences, 2023, 13(19): 10849.
    5.2. Kilpatrick’s Criterion



    :param E_unit_in_MV_m: 击穿场强，单位MV/m
    :return: f——击穿场强对应的频率，unit in MHz
    """
    return 1.643*E_unit_in_MV_m**2 / numpy.exp(-8.5 / E_unit_in_MV_m)
def slac_XXX(f_GHz):
    """
    Ref:
    刘勇涛, 胡银富和盛兴. 《C波段6MeV驻波加速管的仿真设计》. 真空电子技术, 期 2 (2018年): 31～36. https://doi.org/10.16540/j.cnki.cn11-2485/tn.2018.02.05.

    :param f_GHz:
    :return:
    """
    return 220* f_GHz**(1/3)
if __name__ == '__main__':

    plt.ion()

    GHz  = 1e9




    plt.figure()
    Es = numpy.linspace(1,300, 2000)
    lines= plt.plot(Es, Kilpatrick(Es)/1e3)
    plt.xlabel("Surface E-field (MV/m)")
    plt.ylabel("Frequency (GHz)")
    from shapely import LineString
    data  = lines[0].get_data()
    ls_kilpatrick =LineString(numpy.array(data).T)

    freq_to_query  =(
        # 9.3e9#476e6#9.3e9
        5712e6
    )
    ls_query= LineString([[data[0][0],freq_to_query/GHz],
                          [data[0][-1],freq_to_query/GHz]])
    intersection_pts = ls_query.intersection(ls_kilpatrick)
    from shapely.geometry import MultiPoint
    if isinstance(intersection_pts, shapely.geometry.point.Point):
        intersection_pts =MultiPoint( [intersection_pts])
    intersection_pts = numpy.array([ geom.xy for geom in intersection_pts.geoms])[...,0]
    intersection_pts = intersection_pts[numpy.argsort(intersection_pts[:,0])]

    E_queried = intersection_pts[-1,0]
    label = "%.1f MV/m @ %.1f GHz"%(E_queried, freq_to_query/GHz)
    plt.scatter(*intersection_pts[-1],label = label )
    plt.legend()
    from _logging import logger
    logger.info(label)




