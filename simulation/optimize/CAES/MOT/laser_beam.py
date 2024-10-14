# -*- coding: utf-8 -*-
# @Time    : 2024/5/12 12:29
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : laser_beam.py
# @Software: PyCharm
import typing

import numpy


class LaserInfo:
    def __init__(self, k, epsilon, Delta):
        self.k = k  # 波矢
        self.epsilon = epsilon  # 极化，+1表示右手圆极化
        self.Delta = Delta

class LaserData:
    def __init__(self, laser_info, grid, intensity_data):
        self.laser_info = laser_info
        self.intensity_data = intensity_data
        self.grid = grid


class MOField:
    """
    MOField = Magnetic Field + Optical Field (磁光场)
    用于加速计算
    """

    def __init__(self, grid, B_data,laser_datas:typing.List[LaserData] ):
        self.grid = grid
        self.B_data = B_data
        self.laser_datas = laser_datas


# class LaserBeamInMagneticField(LaserBeam):
#     """
#     磁场中的激光束
#     """
#     def __init__(self,k, epsilon, I_interpolator, Delta,
#                  B_interpolator,grid):
#         super(LaserBeamInMagneticField, self).__init__(k, epsilon, I_interpolator, Delta)
#         self.costheta_interpolator =  numpy.dot(B_interpolator, I_interpolator)
def get_costheta_between_wavevector_and_magnetic(k, ):
    pass


class NonDiffractionPlanarSource:
    def __init__(self, source_intensity_interpolator, source_norm_vec, point_on_source):
        self.source_intensity_interpolator = source_intensity_interpolator
        self.source_norm_vec = source_norm_vec
        self.point_on_source = point_on_source  # 源平面上的一点

    def projection_on_source_plane(self, pt_to_be_projected, k: numpy.ndarray):
        return numpy.linalg.pinv(numpy.matrix([
            [*self.source_norm_vec, ],
            [k[1], -k[0], 0],
            [0, k[2], -k[1]],
            [k[2], 0, -k[0]]
        ])) * numpy.matrix(
            [
                numpy.dot(self.source_norm_vec, self.point_on_source),
                k[1] * pt_to_be_projected[0] - k[0] * pt_to_be_projected[1],
                k[2] * pt_to_be_projected[1] - k[1] * pt_to_be_projected[2],
                k[2] * pt_to_be_projected[0] - k[0] * pt_to_be_projected[2]
            ]).T


class NonDiffractionLaserBeam:
    """
    不考虑衍射效应的激光光束
    光强分布是沿着波矢反向投影到光束发射面所得
    """

    def __init__(self, laser_info:LaserInfo, source: NonDiffractionPlanarSource,):
        self.laser_info = laser_info
        self.source = source


    def get_intensity(self, point):
        pt_on_source = self.source.projection_on_source_plane(point, self.laser_info.k)
        return self.source.source_intensity_interpolator(pt_on_source)


if __name__ == '__main__':
    import _grid
    grid = _grid.default_grid
    pt1 = numpy.array([1, 2, 3])
    ps = NonDiffractionPlanarSource(None, numpy.array([1, 2, 1]), numpy.array([0, 0, 0]))
    ps.projection_on_source_plane(pt1, numpy.array([0, 0, 1]))


