# -*- coding: utf-8 -*-
# @Time    : 2025/10/8 17:36
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : COMSOL_B_map_to_GPT_input_files.py
# @Software: PyCharm

import re

import matplotlib
import numpy
import pandas
import scipy.constants as C
from scipy.integrate import solve_ivp

import common
from theory.ebeam_dynamic_in_HPM_device.ebeam_transverse_.ebeam_transverse import \
    AnnularBeamInsideCoaxialDriftWeiYuanZhang
from theory.magnet.expand_near_a_line_for_axisymmetric_B_field_without_source import \
    NoDivNoCurlNoAngularComponentAxisSymmetricFieldExtrapolator
from simulation.task_manager.simulator import df_to_gdf

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

path = r'E:\SharingDirOnIntranet\TTO_01\COMSOL\B_export_inner_only.txt'

def COMSOL_data_to_Dataframe(path):
    return       pandas.read_csv(path, skiprows=9, sep=r'\s+', header=None)

def convert_to_gdf(path=path,    GPT_rundir ="GPT_rundir"):
    df = COMSOL_data_to_Dataframe(path)
    lenunit_in_COMSOL_exported = mm = 1e-3

    df_export_to_GPT = df.copy()
    df_export_to_GPT[[0,1]] *=lenunit_in_COMSOL_exported
    df_export_to_GPT.columns = "r z Br Bz".split(" ")
    df_export_to_GPT = df_export_to_GPT.fillna(0.)


    df_to_gdf(df_export_to_GPT,GPT_rundir + "/B_static_map_2D.gdf",False)

if __name__ == '__main__':

    convert_to_gdf(path)