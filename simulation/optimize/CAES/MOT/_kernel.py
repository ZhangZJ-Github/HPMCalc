# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 21:11
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _kernel.py
# @Software: PyCharm
import _grid
class Field :
    def __init__(self, grid:_grid.Grid3D, values):
        self.grid = grid
        self.values = values
        # self.interpolator =