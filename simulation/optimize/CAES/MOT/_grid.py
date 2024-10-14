# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 19:42
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _grid.py
# @Software: PyCharm
import numpy


class Grid3D:
    def __init__(self, xs, ys, zs):
        # 计算区域大小, unit in meter

        self.xs, self.ys, self.zs = xs, ys, zs
        self.X, self.Y, self.Z = numpy.meshgrid(x, y, z, indexing='ij')


# 计算区域大小, unit in meter
Lx, Ly, Lz = 2.5e-2, 2.5e-2, 2.5e-2

Nx, Ny, Nz = 21, 23, 25

default_grid = Grid3D(numpy.linspace(-Lx / 2, Lx / 2, Nx),
                      numpy.linspace(-Ly / 2, Ly / 2, Ny),
                      numpy.linspace(-Lz / 2, Lz / 2, Nz))

x, y, z = default_grid.xs, default_grid.ys, default_grid.zs
X, Y, Z = default_grid.X, default_grid.Y, default_grid.Z
