# -*- coding: utf-8 -*-
# @Time    : 2025/9/26 12:54
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : process_CST_trajectory_for_shared_EGun.py
# @Software: PyCharm
import os
import re
import cst.results
import matplotlib
import numpy

import common

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas
# proj_path = r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Egun_01.with_B_field.cst"
# proj_path = r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Egun_01.AccStru.with_B_field.cst"
# proj_path = r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Egun_01.AccStru.with_TM.TRK.cst"
proj_path = r"E:\SharingDirOnIntranet\CI-Linac_03_01\CoaxialSource\Egun\Egun.cst"

df = pandas.read_csv(
     os.path.splitext(proj_path)[0]+ r"\Export\3d\Trajectories.txt",
    sep=r'\s+',
    header=None,
    skiprows=6
)
df.columns = re.split(
    r'\s+',
    "posX             posY             posZ             momX             momY             momZ             mass     macro-charge             time       particleID         sourceID    SEEGeneration"
)
cst_proj = cst.results.ProjectFile(proj_path,allow_interactive=True)
key_parid = "particleID"
par_ids = df[key_parid].unique()
par_id_to_plot = par_ids[::100]
# par_id_to_plot = par_ids[::10]

df_to_plot = df[df[key_parid].isin(par_id_to_plot)]
df_to_plot_groupby_parid = df_to_plot.groupby(key_parid)
mm = 1e-3


fig,axs =plt.subplots(3, 1 , sharex= True)
plt.sca(axs[0])
plt.ylabel("r (mm)")

for parid in par_id_to_plot:
    temp_df = df_to_plot_groupby_parid.get_group(parid)

    if 1:
        plt.plot(temp_df['posZ'] / mm,
             (temp_df["posX"] ** 2 + temp_df["posY"] ** 2) ** 0.5 / mm,
                 lw = 1e-1,
                 c = 'r'
             )
        r_beam_center_mm = 65
        dr_beam_tunnel_mm = 6
        plt.axhline(r_beam_center_mm-dr_beam_tunnel_mm /2 ,ls= ":"
                    # alpha = 0.01
                    )
        plt.axhline(r_beam_center_mm+dr_beam_tunnel_mm /2 ,ls= ":"
                    # alpha = 0.01
                    )

    if 0:
        plt.plot(temp_df['posZ'] / mm,
                 temp_df["posX"]  / mm
                 )
        plt.axhspan(-0.5,+0.5,alpha = 0.01)
if 1:
    B_data = numpy.array(cst_proj.get_3d().get_result_item('Tables\\1D Results\\Predefined B-Field (B_exported.txt)_Z (Z)').get_data())
    plt.sca(axs[1])
    plt.plot(B_data[:,0],B_data[:,1])
    plt.ylabel("$B$ (T)")

    if 0:
        B_data_on_axis = numpy.array(cst_proj.get_3d().get_result_item('Tables\\1D Results\\Predefined B-Field (B_exported.txt)_Z (Z)_1').get_data())
        plt.sca(axs[1])
        plt.plot(B_data_on_axis[:,0],B_data_on_axis[:,1])
if 1:
    I_beam_data =numpy.array(cst_proj.get_3d().get_result_item('Tables\\1D Results\\Particle 2D Monitor 1 - total( Current )').get_data())
    plt.sca(axs[2])
    plt.plot(I_beam_data[:,0],I_beam_data[:,1])
    plt.ylabel("$I_b$ (A)")
plt.xlabel("z (mm)")


par_2d_monitor_path =  os.path.splitext(proj_path)[0]+  r'\Export\3d\Particle 2D Monitor 1.txt'
with open(par_2d_monitor_path) as f:
    lines = f.readlines()
my_lines = []
for line in lines[7:]:
    if not line .startswith('%  Plane id'): my_lines.append(line)
from io import  StringIO
df_par2dmonitors = pandas.read_csv(StringIO(''.join(my_lines )),sep = r'\s+',header = None)
df_par2dmonitors.columns = re.split(
    r'\s+',
    "posX             posY             posZ             momX             momY             momZ             mass     macro-charge             time       particleID         sourceID          Current    SEEGeneration"
)
len_eps = 1e-6

df_par2dmonitors["posZ_unit_in_eps"] = df_par2dmonitors['posZ'] // len_eps
temp_df_par2dmonitors_gb_Z = df_par2dmonitors.groupby('posZ_unit_in_eps')
posZ_unit_in_eps = list(temp_df_par2dmonitors_gb_Z.groups.keys())
i = 5
temp_df_par2dmonitors = temp_df_par2dmonitors_gb_Z.get_group(posZ_unit_in_eps[i])
plt.figure()
plt.scatter(temp_df_par2dmonitors['posX'] / mm,temp_df_par2dmonitors['posY'] / mm,s = 0.5)
plt.title("z = %.2f mm"%(posZ_unit_in_eps[i] * len_eps/ mm))
plt.figure()
plt.scatter(temp_df_par2dmonitors['posX'] / mm,temp_df_par2dmonitors['momX'] / temp_df_par2dmonitors['momZ'],s = 0.5)
plt.title("z = %.2f mm"%(posZ_unit_in_eps[i] * len_eps/ mm))



def calculate_nemi_rms(df_,component_name = 'X' # or 'Y'
                       ):
    """
    Ref:

    S. B. van der Geer和M. J. de Loos. 《General Particle Tracer User Manual (Version 3.38)》. 2008年. pp. 97

    Section 3.11.3

    ....
    To calculate the normalized RMS-emittances, first xc, yc, x’c and y’c are calculated by GDFA using:
    ...


    :param df_:
    :return:
    """
    weight = df_["macro-charge"].values
    key_pos = 'pos' + component_name
    key_mom = 'mom' + component_name
    avg_x2 = numpy.average( df_[key_pos].values **2 ,weights=weight)
    avg_xp2 = numpy.average( (df_[key_mom] /df_['momZ'] ).values **2 ,weights=weight)
    avg_x_xp = numpy.average(  df_[key_pos] * (df_[key_mom] /df_['momZ'] ).values  ,weights=weight)
    avg_gamma = numpy.average(   common.gammabeta_to_gamma(df_[key_mom] .values )  ,weights=weight)
    return avg_gamma * (avg_x2 *avg_xp2 - avg_x_xp ** 2 ) ** 0.5

emittances = []
for i,posZ in enumerate( posZ_unit_in_eps):
    df_ =  temp_df_par2dmonitors_gb_Z.get_group(posZ_unit_in_eps[i])
    emittances.append([calculate_nemi_rms(df_,'X'),calculate_nemi_rms(df_,"Y")])

plt.figure()
plt.plot(numpy.array(posZ_unit_in_eps) * len_eps / mm,numpy.array(emittances) / 1e-6,label = ['$\epsilon_{n,x}$','$\epsilon_{n,y}$'])
plt.xlabel("z (mm)")
plt.ylabel("emittance ($\pi$ mm mrad)")
plt.legend()