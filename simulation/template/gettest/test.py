import fld_parser
import par_parser
import grd_parser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # filepath = "E:\\11-18\\1\\Genac10G50keV-1.fld"
    # fld = fld_parser.FLD(filepath)
    filepath = "E:\\11-18\\3\\Genac10G50keV-1.grd"
    grd = grd_parser.GRD(filepath)
    # Bzst_name=' FIELD BZ_ST @LINE_PARTICLE_MOVING$ #1.1'
    # z=grd.ranges[Bzst_name][0]['data'][0]
    # b=grd.ranges[Bzst_name][0]['data'][1]
    # plt.plot(z, b)
    # plt.ylim(0, 1.5)
    # plt.show()
    jdas = grd.ranges[' FIELD_INTEGRAL J.DA @OSYS$AREA,FFT #4.1'][-20:]
    for jda in jdas:
        plt.plot(*jda['data'].values.T)
    plt.show()