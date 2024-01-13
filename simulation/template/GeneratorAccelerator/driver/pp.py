# -*- coding: utf-8 -*-
# @Time    : 2023/11/2 22:06
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : pp.py
# @Software: PyCharm
"""
Codes for Post-Processing
后处理代码
"""
import grd_parser
import matplotlib
import pandas

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

plt.ion()

logdf = pandas.read_csv(r"E:\GeneratorAccelerator\Genac\optmz\Genac20G100keV\粗网格\单独处理\driver.rundir\driver.m2d.log.csv",
                        encoding='gbk')
fig1 = plt.figure()
fig2 = plt.figure()
res_xlsx_extrcav = r'E:\GeneratorAccelerator\Genac\optmz\Genac20G100keV\粗网格\单独处理\driver.rundir/数据处理.extrcav.csv'
res_xlsx_sws1 = r'E:\GeneratorAccelerator\Genac\optmz\Genac20G100keV\粗网格\单独处理\driver.rundir/数据处理.sws1.csv'
extrcav_Ez_FFT_title = ' FIELD E1 @EXTRACTOR1.OBSPT,FFT-#12.2'
Ez_sws1_slot1_title = ' FIELD E1 @SWS1.1.SLOT.OBSPT,FFT-#13.2'
df_extrcav = pandas.DataFrame()
df_sws1 = pandas.DataFrame()
for i in logdf.index[:]:
    grd = grd_parser.GRD(logdf['m2d_path'][i][:-3] + 'grd')
    fig1.gca().plot_AppleGate_diagram()
    df_extrcav['freq/GHz %sws.N%' + '=%d' % (logdf["%sws1.N%"][i])] = grd.obs[extrcav_Ez_FFT_title]['data'][0]
    df_extrcav['Ez Mag. %sws.N%' + '=%d' % (logdf["%sws1.N%"][i])] = grd.obs[extrcav_Ez_FFT_title]['data'][1]
    if Ez_sws1_slot1_title in grd.obs.keys():
        fig2.gca().plot_AppleGate_diagram()
        df_sws1['freq/GHz %sws.N%' + '=%d' % (logdf["%sws1.N%"][i])] = grd.obs[Ez_sws1_slot1_title]['data'][0]
        df_sws1['Ez Mag. %sws.N%' + '=%d' % (logdf["%sws1.N%"][i])] = grd.obs[Ez_sws1_slot1_title]['data'][1]

df_extrcav.to_csv(res_xlsx_extrcav, index=False)
df_sws1.to_csv(res_xlsx_sws1, index=False)

fig1.gca().legend()
fig1.gca().set_title(extrcav_Ez_FFT_title)
fig2.gca().legend()
fig2.gca().set_title(Ez_sws1_slot1_title)
