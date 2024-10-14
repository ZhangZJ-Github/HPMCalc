# -*- coding: utf-8 -*-
# @Time    : 2024/10/11 10:14
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : export_results3.py
# @Software: PyCharm
import enum

import cst.results
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas

import time_dependent_output as tdo
from _logging import logger

matplotlib.use('tkagg')
plt.ion()
cst_proj_path = r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch_1-1.paramsweep.cst"
proj_discharging: cst.results.ProjectFile = cst.results.ProjectFile(cst_proj_path,
                                                                    allow_interactive=True)
proj_charging: cst.results.ProjectFile = cst.results.ProjectFile(
    proj_discharging.filename[:-len("paramsweep.cst")] + "ES.cst",
    allow_interactive=True)
run_ids = proj_discharging.get_3d().get_all_run_ids()
f_ref = 9.3e9
dt = 1 / f_ref / 5.
ts_to_calculate_TD_response = numpy.arange(-20e-9, 0, dt)
initial_signal = (ts_to_calculate_TD_response, numpy.sin(2 * numpy.pi * f_ref * ts_to_calculate_TD_response))
run_id_charging = 0
gamma_3 = numpy.array(
    proj_charging.get_3d().get_result_item('1D Results\\Port Information\\Gamma\\3(1)', run_id_charging).get_data())
nw_charging = tdo.build_network_from_CST_S_data(
    tdo.get_S_parameter_from_CST_proj(proj_charging, "S3,3", run_id_charging),
    tdo.get_S_parameter_from_CST_proj(proj_charging, "S2,3", run_id_charging), )

df_to_plot = pandas.DataFrame()


class ColumnsOf_df_to_plot:
    run_id = "run_id"

    class CSTParameters(enum.Enum):
        GDT_z = "SwitchZoffset"
        GDT_y = "SwitchRoffset"

    class Results(enum.Enum):
        S23_abs_at_f0 = 0
        power_extraction_efficiency_of_minimum_compressor = 1
        theoretical_maximum_power_gain = 2


for run_id in run_ids:
    logger.info("run_id = %d" % run_id)
    try:
        resampled_f = numpy.arange(9e9, 9.6e9, 0.01e9)

        compressor = tdo.Compressor(
            tdo.build_network_from_CST_S_data(tdo.get_S_parameter_from_CST_proj(proj_discharging, "S3,3", run_id),
                                              tdo.get_S_parameter_from_CST_proj(proj_discharging, "S2,3", run_id),  ).interpolate(resampled_f).extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))
                         ,
            nw_charging.interpolate(resampled_f).extrapolate_to_dc(kind="zero", dc_sparam=numpy.zeros((2, 2)))
                         ,
        )

        compressor_deembedded = compressor.embed_with_waveguide(gamma_3, -((300 - 40 - 20)) * 1e-3)
        Dt_MESS = compressor_deembedded.correct_Dt_MESS(f_ref, 10e-9)
        df_i3_charging, df_i3_discharging, df_o33, df_o23 = compressor_deembedded.run(initial_signal, 10e-9, Dt_MESS)
        power_extraction_efficiency_of_minimum_compressor = (df_o23[tdo.key_complex][df_o23[0] > 0].abs() ** 2).max()
        parameters = proj_discharging.get_3d().get_parameter_combination(run_id)

        record = {ColumnsOf_df_to_plot.run_id: run_id}

        for cst_parameter in ColumnsOf_df_to_plot.CSTParameters:
            record[cst_parameter.value] = parameters[cst_parameter.value]
        record.update({
            ColumnsOf_df_to_plot.Results.S23_abs_at_f0.name: numpy.abs(compressor.nw_discharging.interpolate([f_ref]).s[0, 1, 0]),
            ColumnsOf_df_to_plot.Results.power_extraction_efficiency_of_minimum_compressor.name: power_extraction_efficiency_of_minimum_compressor,
            ColumnsOf_df_to_plot.Results.theoretical_maximum_power_gain.name: 1,
        }, )
        df_to_plot = pandas.concat([df_to_plot, pandas.DataFrame([record])], ignore_index=True)

    except ValueError as e:
        logger.info(e)
        continue

def query(zoff, yoff, EPS=0.5, df_to_plot=df_to_plot):
    return df_to_plot[(numpy.abs(df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_z.value] - zoff) < EPS) & (numpy.abs(df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_y.value] - yoff) < EPS)]

from scipy.spatial import Delaunay

tri =Delaunay(df_to_plot[[ColumnsOf_df_to_plot.CSTParameters.GDT_z.value,ColumnsOf_df_to_plot.CSTParameters.GDT_y.value]].values)

plt.figure()
cf = plt.tricontourf(df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_z.value],
                df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_y.value],
                tri.simplices,
                df_to_plot[ColumnsOf_df_to_plot.Results.power_extraction_efficiency_of_minimum_compressor.name],)

plt.triplot(df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_z.value],
            df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_y.value],tri.simplices,lw = 0.5)
plt.gcf().colorbar(cf)
plt.figure()
cf = plt.tricontourf(df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_z.value],
                df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_y.value],
                tri.simplices,
                numpy.abs(df_to_plot[ColumnsOf_df_to_plot.Results.S23_abs_at_f0.name]),)

plt.triplot(df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_z.value],
            df_to_plot[ColumnsOf_df_to_plot.CSTParameters.GDT_y.value],tri.simplices,lw = 0.5)
plt.gcf().colorbar(cf)
