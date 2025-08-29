# -*- coding: utf-8 -*-
# @Time    : 2025/7/20 22:01
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : mode_coverter_optimize.py
# @Software: PyCharm
import matplotlib
import scipy.interpolate

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy
from theory.ModeConverter.CoaxTEMtoRectTE10.network_with_medias import NetWorkwithMedias
import skrf
from _logging import logger
from task_manager.recorders import ProgressRecorder
from scipy.optimize import minimize

if __name__ == '__main__':

    CoaxWithSuppRods = NetWorkwithMedias.from_CST(
        r"E:\SharingDirOnIntranet\TTO_01\CST\ModeConverter\CoaxToRect\CoaxWithSuppRods.cst")
    CoaxWithSuppRods.nw.name = "CoaxWithSuppRods"
    CoaxToRect = NetWorkwithMedias.from_CST(
        r"E:\SharingDirOnIntranet\TTO_01\CST\ModeConverter\CoaxToRect\CoaxToRect.cst")
    CoaxToRect.nw.name = "CoaxToRect"
    resampling_frequency = CoaxToRect.nw.frequency

    f_center = 9.3e9
    Df = 0.2e9
    f_to_evaluate_avg_Prefl_coeff = numpy.linspace(f_center - Df, f_center + Df, 200)


    def get_total_power_reflection_coeff(args,
                                         # data_to_record: dict
                                         **kwargs
                                         ):
        line1_length_mm = args[0]
        data_to_record = {}
        nw_HOM_coax_line = [
            med.line(line1_length_mm, unit='mm', name="Port Mode %d" % (i + 1)).interpolate(resampling_frequency)
            for i, med in enumerate(CoaxToRect.medias)]

        connexion = [
                        [(CoaxWithSuppRods.nw, 5 + i), (line, 0)] for i, line in enumerate(nw_HOM_coax_line)
                    ] + [
                        [(line, 1), (CoaxToRect.nw, 0 + i)] for i, line in enumerate(nw_HOM_coax_line)
                    ] + [
                        [(skrf.Circuit.Port(frequency=resampling_frequency, name="Port 1(%d)" % (i + 1)), 0),
                         (CoaxWithSuppRods.nw, 0 + i)]
                        for i in range(len(nw_HOM_coax_line))
                    ]

        cir = skrf.Circuit(connexion, name="Mode Converter")
        # plt.figure()
        if 0:
            cir.plot_graph(network_labels=True, network_fontsize=15,
                           port_labels=True, port_fontsize=15,
                           edge_labels=True, edge_fontsize=10)
        # cir.network.plot_s_mag()
        nw = cir.network
        power_total_reflection_coeff = numpy.abs(numpy.sum(numpy.abs(numpy.abs(nw.s[:, :, 0])) ** 2, axis=1))

        Prefl_coeff_interp = scipy.interpolate.interp1d(resampling_frequency.f, power_total_reflection_coeff)

        avg_Prefl_coeff = numpy.nanmean(Prefl_coeff_interp(f_to_evaluate_avg_Prefl_coeff))

        if kwargs.get("plot_Sparam", False):
            plt.figure()
            for i, _ in enumerate(nw_HOM_coax_line):
                plt.plot(resampling_frequency.f / 1e9,
                         numpy.abs(nw.s[:, i, 0]), label="$S_{1(%d),1(1)}$" % (i + 1))

            plt.plot(resampling_frequency.f / 1e9, power_total_reflection_coeff ** 0.5,
                     label="total", lw=3)
            plt.axvspan(f_to_evaluate_avg_Prefl_coeff[0] / 1e9, f_to_evaluate_avg_Prefl_coeff[-1] / 1e9, alpha=0.1)
            plt.legend()
            plt.ylim(0, 1)
        data_to_record.update({
            "args":args,
            "line1_length_mm":line1_length_mm,
            "avg_Prefl_coeff":avg_Prefl_coeff,
        })
        pr.write_parameter_and_result_to_csv(data_to_record)

        return avg_Prefl_coeff


    pr = ProgressRecorder()

    min_res = minimize(
        get_total_power_reflection_coeff,
        numpy.array(
            [7.0]
        ),
        tol=0.001,
        bounds=
        [
            [-20.,40.],
        ],
        options={"maxiter": 1000}, method="Nelder-Mead",
        # callback=
    )
    logger.info("min_res.success = %s" % min_res.success)
    Prefl_coeff = get_total_power_reflection_coeff(min_res.x, plot_Sparam=True)
