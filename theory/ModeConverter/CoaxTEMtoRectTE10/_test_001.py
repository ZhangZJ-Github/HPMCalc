# -*- coding: utf-8 -*-
# @Time    : 2025/7/18 12:06
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test_001.py
# @Software: PyCharm
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy
from theory.ModeConverter.CoaxTEMtoRectTE10.network_with_medias import NetWorkwithMedias
import skrf

if __name__ == '__main__':

    CoaxWithSuppRods = NetWorkwithMedias.from_CST(r"E:\SharingDirOnIntranet\TTO_01\CST\ModeConverter\CoaxToRect\CoaxWithSuppRods.cst")
    CoaxWithSuppRods.nw.name = "CoaxWithSuppRods"
    CoaxToRect = NetWorkwithMedias.from_CST(r"E:\SharingDirOnIntranet\TTO_01\CST\ModeConverter\CoaxToRect\CoaxToRect.cst")
    CoaxToRect.nw.name = "CoaxToRect"
    line1_length_mm = -10#20
    resampling_frequency  = CoaxToRect.nw.frequency
    nw_HOM_coax_line = [ med.line(line1_length_mm,unit = 'mm',name = "Port Mode %d"%(i+1)).interpolate(resampling_frequency)
                         for i, med in enumerate( CoaxToRect.medias)]


    connexion = [
        [(CoaxWithSuppRods.nw, 5 + i ),(line, 0)]for i,line in enumerate(nw_HOM_coax_line)
    ]+[
        [(line, 1),(CoaxToRect.nw, 0+ i)] for i,line in enumerate(nw_HOM_coax_line)
    ] + [
        [(skrf.Circuit.Port(frequency= resampling_frequency, name = "Port 1(%d)"%( i + 1)),0),(CoaxWithSuppRods.nw,0 + i )]
        for i in range(len(nw_HOM_coax_line))
    ]

    cir = skrf.Circuit(connexion,name = "Mode Converter")
    # plt.figure()
    if 0:
        cir.plot_graph(network_labels=True, network_fontsize=15,
                   port_labels=True, port_fontsize=15,
                  edge_labels=True, edge_fontsize=10)
    plt.figure()
    # cir.network.plot_s_mag()
    nw = cir.network
    for i,_ in enumerate(nw_HOM_coax_line):
        plt.plot(resampling_frequency.f /1e9,
            numpy.abs(nw.s[:,i,0]),label = "$S_{1(%d),1(1)}$"%(i +1 ))

    plt.plot(resampling_frequency.f /1e9,numpy.sum(numpy.abs(numpy.abs(nw.s[:,:,0]))**2 ,axis = 1)**0.5,label = "total",lw = 3)
    plt.legend()
    plt.ylim(0, 1)
    # nw_coax_with_suprods =  DefinedGammaZ0()

    from task_manager.recorders import ProgressRecorder

    pr = ProgressRecorder()
    pr.write_parameter_and_result_to_csv({"a": 0, "b": 1})
