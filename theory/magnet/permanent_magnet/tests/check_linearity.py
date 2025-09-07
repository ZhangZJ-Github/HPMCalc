# -*- coding: utf-8 -*-
# @Time    : 2025/9/2 18:06
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : check_linearity.py
# @Software: PyCharm
# 检查仅包含永磁块的模拟结果是否支持线性叠加

import cst.results
import numpy
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
class Permanent_CST_Project:
    def __init__(self,proj_path):
        self.proj_path  = proj_path
        self.proj =  cst.results.ProjectFile(proj_path,
                                             allow_interactive=True)
        self.Bz_data =  numpy.array(self.proj.get_3d().get_result_item('Tables\\1D Results\\B-Field (Ms)_Z (Z)').get_data())
proj_tot = Permanent_CST_Project(r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Magnet_03.cst",)
proj_01 = Permanent_CST_Project(r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Magnet_03.01.cst",)
proj_02 = Permanent_CST_Project(r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Magnet_03.02.cst",)
proj_03 = Permanent_CST_Project(r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Magnet_03.03.cst",)



plt.figure()
plt.plot(proj_tot.Bz_data[:,0],proj_tot.Bz_data[:,1],
         )

plt.plot(
proj_tot.Bz_data[:,0],
    proj_01.Bz_data[:,1] +
    proj_02.Bz_data[:,1] +
    proj_03.Bz_data[:,1]

)
