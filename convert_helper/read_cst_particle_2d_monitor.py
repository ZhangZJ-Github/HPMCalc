# -*- coding: utf-8 -*-
# @Time    : 2025/12/11 11:53
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : read_cst_particle_2d_monitor.py
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
import scipy.constants as C
from simulation.task_manager.simulator import df_to_gdf
from io import StringIO

from _logging import logger

# proj_path = r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Egun_01.with_B_field.cst"
# proj_path = r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Egun_01.AccStru.with_B_field.cst"
# proj_path = r"E:\SharingDirOnIntranet\TTO_01\CST\Eguns\EGunForCoaxialSource\GyroLike\Egun_01.AccStru.with_TM.TRK.cst"
proj_path = r"E:\SharingDirOnIntranet\CI-Linac_03_01\CoaxialSource\Egun\Egun.cst"
df = pandas.read_csv(
    os.path.splitext(proj_path)[0] + r"\Export\3d\Trajectories.txt",
    sep=r'\s+',
    header=None,
    skiprows=6
)
# df[2] = df[2].astype(float)
df.columns = re.split(
    r'\s+',
    "posX             posY             posZ             momX             momY             momZ             mass     macro-charge             time       particleID         sourceID    SEEGeneration"
)
cst_proj = cst.results.ProjectFile(proj_path, allow_interactive=True)
key_parid = "particleID"
par_ids = df[key_parid].unique()
# par_id_to_plot = par_ids[::200]
par_id_to_plot = par_ids[::10]

df_to_plot = df[df[key_parid].isin(par_id_to_plot)]
df_to_plot_groupby_parid = df_to_plot.groupby(key_parid)

mm = 1e-3
fig, axs = plt.subplots(2, 1, sharex=True)
plt.sca(axs[0])
for parid in par_id_to_plot:
    temp_df = df_to_plot_groupby_parid.get_group(parid)
    if 0:
        plt.plot(temp_df['posZ'] / mm,
                 (temp_df["posX"] ** 2 + temp_df["posY"] ** 2) ** 0.5 / mm
                 )
    if 1:
        plt.plot(temp_df['posZ'] / mm,
                 temp_df["posX"] / mm
                 )
        plt.axhspan(-0.5, +0.5, alpha=0.01)

plt.figure()
for parid in par_id_to_plot:
    temp_df = df_to_plot_groupby_parid.get_group(parid)

    plt.plot(temp_df['posZ'] / mm,
             (common.gammabeta_to_gamma(
                 (temp_df["momX"] ** 2 +
                  temp_df["momY"] ** 2 +
                  temp_df["momZ"] ** 2) ** 0.5) - 1) * C.m_e * C.c ** 2 / C.eV / 1e3
             )
plt.xlabel("z (mm)")
plt.ylabel("KE (keV)")
# plt.axhspan(-0.5, +0.5, alpha=0.01)
if 0:
    B_data = numpy.array(
        cst_proj.get_3d().get_result_item('Tables\\1D Results\\Predefined B-Field (B_exported.txt)_Z (Z)').get_data())
    plt.sca(axs[1])
    plt.plot(B_data[:, 0], B_data[:, 1])

    B_data_on_axis = numpy.array(
        cst_proj.get_3d().get_result_item('Tables\\1D Results\\Predefined B-Field (B_exported.txt)_Z (Z)_1').get_data())
    plt.sca(axs[1])
    plt.plot(B_data_on_axis[:, 0], B_data_on_axis[:, 1])
par_2d_monitor_path = os.path.splitext(proj_path)[0] + r'\Export\3d\Particle 2D Monitor 1.txt'
with open(par_2d_monitor_path) as f:
    lines = f.readlines()
my_lines = []
for line in lines[7:]:
    if not line.startswith('%  Plane id'): my_lines.append(line)
df_par2dmonitors = pandas.read_csv(StringIO(''.join(my_lines)), sep=r'\s+', header=None)
df_par2dmonitors.columns = re.split(
    r'\s+',
    "posX             posY             posZ             momX             momY             momZ             mass     macro-charge             time       particleID         sourceID          Current    SEEGeneration"
)
len_eps = 1e-9

df_par2dmonitors["posZ_unit_in_eps"] = df_par2dmonitors['posZ'] // len_eps
temp_df_par2dmonitors_gb_Z = df_par2dmonitors.groupby('posZ_unit_in_eps')
posZ_unit_in_eps = list(temp_df_par2dmonitors_gb_Z.groups.keys())
i = 4  # 0
temp_df_par2dmonitors = temp_df_par2dmonitors_gb_Z.get_group(posZ_unit_in_eps[i])


def particle_2d_data_to_MAGIC_EGUN_input(temp_df_par2dmonitors, N_samples=250,
                                         EGUN_in_filename="dfm_3.in"
                                         ):
    """
    将CST TRK 导出的2D monitor粒子数据转化为MAGIC可识别的输入文件
    :param temp_df_par2dmonitors:
    :param N_samples:
    :param EGUN_in_filename:
    :return:
    """
    df_to_MAGIC_EGUN_input = pandas.DataFrame(  # dtype=float
    )
    temp_df_par2dmonitors.loc[:, "posR"] = (temp_df_par2dmonitors['posX'] ** 2 + temp_df_par2dmonitors[
        'posY'] ** 2) ** .5
    # arr_er = numpy.column_stack([temp_df_par2dmonitors['posX'],temp_df_par2dmonitors['posY']],) / temp_df_par2dmonitors.loc[:,"posR"].values.reshape(-1,1)
    arr_theta = numpy.arctan2(temp_df_par2dmonitors['posY'], temp_df_par2dmonitors['posX'], )
    temp_df_par2dmonitors.loc[:, "momR"] = (
                temp_df_par2dmonitors['momX'] * numpy.cos(arr_theta) + temp_df_par2dmonitors['momY'] * numpy.sin(
            arr_theta))
    temp_df_par2dmonitors.loc[:, "momTheta"] = (
                -temp_df_par2dmonitors['momX'] * numpy.sin(arr_theta) + temp_df_par2dmonitors['momY'] * numpy.cos(
            arr_theta))

    temp_df_par2dmonitors_filtered_ = temp_df_par2dmonitors[temp_df_par2dmonitors['momZ'] > 0].reset_index()  # 过滤掉逆流的粒子
    total_forward_beam_current = temp_df_par2dmonitors_filtered_['Current'].sum()
    # N_samples = 100
    temp_df_par2dmonitors_filtered = temp_df_par2dmonitors_filtered_.sample(N_samples).reset_index()
    logger.info("total beam current = %.2f A" % (total_forward_beam_current))
    if 1:temp_df_par2dmonitors_filtered['momZ']*= (
                                                          # 53/48
        53.5/53

                                                              ) **0.5
    if 0:
        temp_df_par2dmonitors_filtered['momX']*= 0
        temp_df_par2dmonitors_filtered['momY']*= 0


    df_to_MAGIC_EGUN_input["n"] = numpy.array(range(temp_df_par2dmonitors_filtered.shape[0])) + 1
    df_to_MAGIC_EGUN_input['t'] = 0.
    UNIT_IN = 0.0053  # .in文件中的参数
    df_to_MAGIC_EGUN_input['grid_R'] = (
            temp_df_par2dmonitors_filtered['posR'] / (
            UNIT_IN * C.inch)).values
    df_to_MAGIC_EGUN_input['Z'] = (temp_df_par2dmonitors_filtered['posZ']).values
    logger.info("Export z = %.2f mm" % (temp_df_par2dmonitors_filtered['posZ'].mean() / mm))
    df_to_MAGIC_EGUN_input['KE (eV)'] = (common.beta_to_gamma(common.p_to_v(
        ((temp_df_par2dmonitors_filtered['momX'] ** 2 + temp_df_par2dmonitors_filtered['momY'] ** 2 +
          temp_df_par2dmonitors_filtered['momZ'] ** 2) ** 0.5).values * C.c,
        1.0
    ) / C.c) - 1) * C.m_e * C.c ** 2 / C.eV
    # theta = numpy.arctan2(temp_df_par2dmonitors_filtered['posX'].values,
    #                       temp_df_par2dmonitors_filtered['posY'].values, )

    df_to_MAGIC_EGUN_input["RayPa"] = numpy.arctan2(
    temp_df_par2dmonitors_filtered['momR'],
        temp_df_par2dmonitors_filtered['momZ']
    )
    total_forward_beam_current_of_sampled_traj = temp_df_par2dmonitors_filtered['Current'].sum()
    logger.info("total_forward_beam_current_of_sampled_traj=%.2f A" % total_forward_beam_current_of_sampled_traj)
    beam_current_factor = 6.283224749190616e-06
    key_norm_rayI = "RayI/factor"
    df_to_MAGIC_EGUN_input[key_norm_rayI] = (
            (temp_df_par2dmonitors_filtered['Current'] / total_forward_beam_current_of_sampled_traj)
            * total_forward_beam_current / beam_current_factor)
    logger.info("Beam current sum (after sampled) = %.4f A" % (
            df_to_MAGIC_EGUN_input[key_norm_rayI].sum() * beam_current_factor))
    df_to_MAGIC_EGUN_input["RayTa"] = numpy.arctan2(
    temp_df_par2dmonitors_filtered['momTheta'],
        temp_df_par2dmonitors_filtered['momZ']
    )
    df_to_MAGIC_EGUN_input["Unknown"] = 1.0
    plt.figure()
    plt.scatter(numpy.zeros(len(temp_df_par2dmonitors_filtered)),temp_df_par2dmonitors_filtered['posR'])

    header = \
        """dfm_3 stage 3 of dfm of 5-21-99
         &INPUT1
          RLIM=35,   ZLIM=1000,             
          POTN=5,    POT(5)=1E+5,
          MI=3,      SX=7.50,   SY = 0.30,    SCALE='   ',
          PASS=8,    XR=0.9950, TYME=10000.0, ERROR=0.10E-2,
          POIS=0,    LSTPOT=0,  LSTMAG=0,     LSTBND=0,
          CSYS=2,    INTPA = 2,
         &END
            0    0    0      0.0000000  0.0000000
          888
         &INPUTA
         COLUMNS=1234,
         &END
         &INPUT5
          START ='CARDS', ZO=-820,      SKAL=1.00,
          MAXRAY =-%d,   IPHI=184,     UNITIN=%.4f,
          NS=80,          SAVE=0,       STEP=0.1,
          AV=78,          AVR=1.0,      SPC=1.0,
          ZDOTEQ=0.10,    MAGMLT=  1.0, IPBP=184, %d,
         &END""" % (df_to_MAGIC_EGUN_input.shape[0], UNIT_IN, df_to_MAGIC_EGUN_input.shape[0])
    text = "%s\n%s" % (header, df_to_MAGIC_EGUN_input.to_string(index=False, header=None))
    df_to_MAGIC_EGUN_input.to_csv("df_to_MAGIC_EGUN_input.csv", index=False, )
    with open(EGUN_in_filename, 'w') as f:
        f.write(text, )
    logger.info("写入了%s" % EGUN_in_filename)
    return text,EGUN_in_filename


text,EGUN_in_filename = particle_2d_data_to_MAGIC_EGUN_input(temp_df_par2dmonitors, 350)
os.system("code %s"%EGUN_in_filename)

plt.figure()
plt.scatter(temp_df_par2dmonitors['posX'] / mm, temp_df_par2dmonitors['posY'] / mm, s=0.5)
plt.title("z = %.2f mm" % (posZ_unit_in_eps[i] * len_eps / mm))
plt.figure()
plt.scatter(temp_df_par2dmonitors['posX'] / mm, temp_df_par2dmonitors['momX'] / temp_df_par2dmonitors['momZ'] / 1e-3,
            s=0.5)
plt.xlabel("x (mm)")
plt.ylabel("x' (mrad)")
plt.title("z = %.2f mm" % (posZ_unit_in_eps[i] * len_eps / mm))

plt.figure()
plt.scatter(temp_df_par2dmonitors['posR'] / mm, temp_df_par2dmonitors['momR']  *C.c,
            s=0.5)
plt.xlabel("r (mm)")
plt.ylabel(r"$\gamma \beta_r c$ (m/s)")
plt.title("z = %.2f mm" % (posZ_unit_in_eps[i] * len_eps / mm))

df_par2dmonitor_export_to_GPT = pandas.DataFrame(
    temp_df_par2dmonitors[['posX', 'posY', 'posZ', 'momX', 'momY', 'momZ', "mass", "time"]].values,
    columns='x y z GBx GBy GBz m t'.split(' '))
df_par2dmonitor_export_to_GPT['nmacro'] = numpy.abs(temp_df_par2dmonitors['macro-charge'].values)

df_par2dmonitor_export_to_GPT['q'] = - C.e

freq = 9.3e9
I_beam = 300  # 0.2
dt = 1 / freq / 100
_ts = numpy.arange(0, 300e-3 / (C.c) * 2, dt)
len_original_df_par2dmonitor_export_to_GPT = df_par2dmonitor_export_to_GPT.shape[0]
N_samples = 5000

df_par2dmonitor_export_to_GPT = pandas.DataFrame(df_par2dmonitor_export_to_GPT.values.repeat(len(_ts), axis=0),
                                                 columns=df_par2dmonitor_export_to_GPT.columns)
df_par2dmonitor_export_to_GPT['t'] += numpy.tile(_ts, len_original_df_par2dmonitor_export_to_GPT)
df_par2dmonitor_export_to_GPT = df_par2dmonitor_export_to_GPT.sample(n=N_samples, random_state=100)
df_par2dmonitor_export_to_GPT['nmacro'] = numpy.abs(I_beam * (_ts[-1] - _ts[0]) / C.e
                                                    # / len(df_par2dmonitor_export_to_GPT)
                                                    ) * (
                                                  df_par2dmonitor_export_to_GPT['nmacro'] /
                                                  df_par2dmonitor_export_to_GPT['nmacro'].sum()
                                          )
GPT_rundir = "GPT_rundir"
aaaaa
df_to_gdf(df_par2dmonitor_export_to_GPT, GPT_rundir + "/par_input.gdf", False)

from scipy.interpolate import RegularGridInterpolator

Es_df = pandas.read_csv(
    os.path.splitext(proj_path)[0] + r"\Export\3d\E-Field [Es]_2d_plane.txt",
    sep=r'\s+',
    header=None,
    skiprows=3
)
# Es_interpolator = LinearNDInterpolator(Es_df[[0,1,2]].values, Es_df[[3,4,5]].values, fill_value=0.0)
Es_df[[0, 1, 2, ]] *= mm
# _key_Er ='Er'
# Es_df[_key_Er] = Es_df[4]
# _filter = (Es_df[1]< 0 )
# Es_df[_key_Er][_filter] *=-1

_ys, _zs = (Es_df[1] / len_eps).astype(int).unique() * len_eps, (Es_df[2] / len_eps).astype(int).unique() * len_eps,
Es_interpolator = RegularGridInterpolator((_zs, _ys,), Es_df[[5, 4]].values.reshape((_zs.size, _ys.size, 2)),
                                          bounds_error=False, fill_value=0.)
# _rs =  numpy.abs(_ys)
# _indexes_of_rs_sorted =numpy.argsort(_rs)
# _rs_sorted= _rs[_indexes_of_rs_sorted]
_R, _Z = numpy.meshgrid(numpy.sort(numpy.unique((numpy.abs(_ys) / len_eps).astype(int))) * len_eps, _zs)
_Estatic = Es_interpolator((_Z, _R))
E_static_export_to_GPT = pandas.DataFrame({
    "z": _Z.ravel(),
    "r": _R.ravel(),
    "Ez": _Estatic[..., 0].ravel(),
    "Er": _Estatic[..., 1].ravel(),
})

df_to_gdf(E_static_export_to_GPT, GPT_rundir + "/E_static_map.gdf", False)

if 0:
    plt.figure()
    vmax = numpy.abs(_Estatic[..., 0]).max()
    cf = plt.contourf(_Z / mm, _R / mm, _Estatic[..., 0], cmap='jet', levels=numpy.linspace(-vmax, vmax, 101))
    plt.gca().set_aspect('equal')
    plt.colorbar(cf)

    plt.figure()
    vmax = numpy.abs(_Estatic[..., 1]).max()
    cf = plt.contourf(_Z / mm, _R / mm, _Estatic[..., 1], cmap='jet', levels=numpy.linspace(-vmax, vmax, 101))
    plt.gca().set_aspect('equal')
    plt.colorbar(cf)


def calculate_nemi_rms(df_, component_name='X'  # or 'Y'
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
    avg_x2 = numpy.average(df_[key_pos].values ** 2, weights=weight)
    avg_xp2 = numpy.average((df_[key_mom] / df_['momZ']).values ** 2, weights=weight)
    avg_x_xp = numpy.average(df_[key_pos] * (df_[key_mom] / df_['momZ']).values, weights=weight)
    avg_gamma = numpy.average(common.gammabeta_to_gamma(df_[key_mom].values), weights=weight)
    return avg_gamma * (avg_x2 * avg_xp2 - avg_x_xp ** 2) ** 0.5


emittances = []
for i, posZ in enumerate(posZ_unit_in_eps):
    df_ = temp_df_par2dmonitors_gb_Z.get_group(posZ_unit_in_eps[i])
    emittances.append([calculate_nemi_rms(df_, 'X'), calculate_nemi_rms(df_, "Y")])

plt.figure()
plt.plot(numpy.array(posZ_unit_in_eps) * len_eps / mm, numpy.array(emittances) / 1e-6,
         label=['$\epsilon_{n,x}$', '$\epsilon_{n,y}$'])
plt.xlabel("z (mm)")
plt.ylabel("emittance ($\pi$ mm mrad)")
plt.legend()
