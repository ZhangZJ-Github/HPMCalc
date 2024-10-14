# -*- coding: utf-8 -*-
# @Time    : 2024/9/27 10:49
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : CST_parametric_sweep.py
# @Software: PyCharm
import enum
import time

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import scipy.constants as C
from scipy.fft import ifft
from scipy.optimize import curve_fit
from shapely.geometry import LineString

from scipy.spatial import Delaunay
import cst.interface


from refinement import  refine_mesh
from scipy.interpolate import LinearNDInterpolator,interp1d
from _logging import logger



matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.ion()

cst_proj_path =  r"E:\CSTprojects\rfCompressor\cascadedHT\SES_switch.TapperedSwitchCav.3.paramsweep.cst"
cst_proj:cst.results.ProjectFile = cst.results.ProjectFile(
        cst_proj_path,
        allow_interactive=True)
all_runids = cst_proj.get_3d().get_all_run_ids()
f0= 9.3e9
GHz = 1e9

df_to_plot = pandas.DataFrame()

class ParameterNames(enum.Enum):
        GDT_y=0
        GDT_z =1

class ResultNames (enum.Enum):
        run_id = 2

        S23_abs_at_f0 = 0
        S23_df = 1
all_runids.remove(0)
for run_id in all_runids:
    try:
        S23_data  = numpy.array(cst_proj.get_3d().get_result_item( '1D Results\\S-Parameters\\S2,3',run_id).get_data())
        S23_abs_at_f0 = numpy.abs(numpy.interp( (f0/GHz) , S23_data[:, 0].real,S23_data[:,1]))
        parameters =cst_proj.get_3d().get_parameter_combination(run_id)
        res = {}
        for parname in ParameterNames:
                res[parname.name] = parameters[parname.name]
        res[ResultNames. S23_abs_at_f0.name] = S23_abs_at_f0
        res[ResultNames.S23_df.name ] = S23_data
        res[ResultNames.run_id.name ] = run_id
        df_to_plot = pandas.concat([df_to_plot, pandas.DataFrame([res])],ignore_index=True)
    except ValueError as e:
        logger.info(e)
        continue
def query(zoff, yoff, EPS=0.5, df_to_plot=df_to_plot):
    return df_to_plot[(numpy.abs(df_to_plot[ParameterNames.GDT_z.name] - zoff) < EPS) & (numpy.abs(df_to_plot[ParameterNames.GDT_y.name] - yoff) < EPS)]

interpolator = LinearNDInterpolator(df_to_plot[[ParameterNames.GDT_z.name,ParameterNames.GDT_y.name,] ].values,df_to_plot[ResultNames.S23_abs_at_f0.name].values,)
def min_and_max (data):return min(data),max(data)
GDT_Z,GDT_Y = numpy.meshgrid (numpy.linspace( * min_and_max(
        df_to_plot[ParameterNames.GDT_z.name]),10),
numpy.linspace(*min_and_max(
        df_to_plot[ParameterNames.GDT_y.name]), 10)
)



# def get_finest_mesh(pts_estimated:numpy.ndarray, min_delta:numpy.ndarray ):
#         """
#
#         :param pts_estimated: shape (N_params, N_points)
#         :param min_delta: shape (N_params,)
#         :return:
#         """
#         return numpy.meshgrid(*[numpy.arange(pts_estimated[i] .min(),pts_estimated[i] .max() + min_delta[i],min_delta[i],) for i in range(pts_estimated.shape[0])])
#
# GDT_Z_finest_mesh, GDT_Y_finest_mesh= get_finest_mesh(df_to_plot[[ParameterNames.GDT_z.name,ParameterNames.GDT_y.name,] ].values.T,numpy.ones(2,)*1.0)
#
# plt.scatter(GDT_Z_finest_mesh,GDT_Y_finest_mesh,s = 1)


tri_mesh  = Delaunay(df_to_plot[[ParameterNames.GDT_z.name,ParameterNames.GDT_y.name,] ].values,)



def build_S_parameter_interpolator():
        fs =   df_to_plot[ResultNames.S23_df.name][0][:,0].real
        li=  LinearNDInterpolator(df_to_plot[[ParameterNames.GDT_z.name,ParameterNames.GDT_y.name,] ].values,
                [arr[:,1]  for arr in                                       df_to_plot[ResultNames.S23_df.name]],)

        return lambda x,y,f: interp1d(fs,  li((x,y)), )(f)

S_parameter_interpolator = build_S_parameter_interpolator()
S_parameter_interpolator([285.5,290],[19.88,18],[9.3,9.4])
f_target = 9.3
tri_mesh_new,new_parametric_space_pts = refine_mesh(tri_mesh, S_parameter_interpolator,f_target=f_target)
new_parametric_space_pts = new_parametric_space_pts[:1]
tri_mesh_new = Delaunay(numpy.array((*tri_mesh.points,*new_parametric_space_pts)))


plt.figure()
# cf = plt.contourf(GDT_Z,GDT_Y, interpolator(GDT_Z,GDT_Y),levels = numpy.linspace(0, 1, 15))
cf=  plt.tricontourf(*tri_mesh.points.T,df_to_plot[ResultNames.S23_abs_at_f0.name].values   ,tri_mesh.simplices,levels = numpy.linspace(0, 1, 15))
plt.scatter(*df_to_plot[[ParameterNames.GDT_z.name,ParameterNames.GDT_y.name,] ].values.T,s = 1)
plt.colorbar(cf)
plt.triplot(*tri_mesh.points.T,tri_mesh.simplices,c = 'k',alpha = 0.5,lw = 0.5)
new_pts_scatter = plt.scatter(*new_parametric_space_pts.T,s = 2)
new_mesh_triplot = plt.triplot(*tri_mesh_new.points.T,tri_mesh_new.simplices,c = 'r',alpha = 0.5,lw = 0.5)
i_max_S21 =  numpy.argmax(df_to_plot[ResultNames.S23_abs_at_f0.name])
msg_about_max_S = "Maximum $|S_{2,1}|$ = %.2f at (%.2f, %.2f)"%(
    df_to_plot[ResultNames.S23_abs_at_f0.name][i_max_S21],
    df_to_plot[ParameterNames.GDT_z.name][i_max_S21],
    df_to_plot[ParameterNames.GDT_y.name][i_max_S21],)
plt.title(msg_about_max_S)
plt.gca().set_aspect('equal')
logger.info(msg_about_max_S)
logger.info(tri_mesh.points.shape)
logger.info(tri_mesh_new.points.shape)
logger.info("新增模拟数量：%d"%(new_parametric_space_pts.shape[0]))
def update_axes(  ):
    #ax.scatter(new_pt[0],new_pt[1],c = "blue",s = 1)
    global  new_pts_scatter,new_mesh_triplot,tri_mesh_new,new_parametric_space_pts

    new_pts_scatter .remove()
    new_pts_scatter = plt.scatter(*new_parametric_space_pts.T, marker = '*',
                                  s=2)
    new_mesh_triplot[0].remove()
    new_mesh_triplot = plt.triplot(*tri_mesh_new.points.T, tri_mesh_new.simplices, c='r', alpha=0.5, lw=0.5)
    # plt.gca().draw()



def delete_pt(pt_lists,  pt_to_remove:numpy.ndarray):
    """

    :param pt_lists: shape (N, 2)
    :param pt_to_remove: shape (2,)
    :return: remained pts, removed pts
    """
    EPS = 0.5
    idx = ((pt_lists - pt_to_remove)**2).sum(axis = 1)**0.5 > EPS


    return  pt_lists[idx],  pt_lists[~idx]

ax = plt.gca()
from matplotlib.backend_bases import  MouseEvent,MouseButton
def on_click(event):
    global new_parametric_space_pts,tri_mesh_new
    # logger.info(event.inaxes)
    if event.inaxes is None:
        return
    # logger.info(event.button)
    new_pt = numpy.array([[event.xdata, event.ydata]])
    if event.button == MouseButton.LEFT:
        logger.info("Left click at ({:.2f}, {:.2f}) in '{}', this point is added".format(
            *new_pt[0], event.inaxes.get_title()))

        new_parametric_space_pts = numpy.array([*new_pt,*new_parametric_space_pts])
        tri_mesh_new = Delaunay((*tri_mesh.points,*new_parametric_space_pts))
        update_axes( )

    elif  event.button == MouseButton.RIGHT:
        new_parametric_space_pts ,deleted_pts=delete_pt(new_parametric_space_pts, new_pt[0])# numpy.array([*new_pt,*new_parametric_space_pts])
        tri_mesh_new = Delaunay((*tri_mesh.points,*new_parametric_space_pts))

        logger.info("Right click at (%.2f, %.2f), points near it are deleted (%s)"%(
            *new_pt[0], deleted_pts ))
        update_axes( )
    elif event .button == MouseButton.MIDDLE:
        EPS = 0.1
        logger.info("\n%s" % (df_to_plot[
            ((df_to_plot[[ParameterNames.GDT_z.name, ParameterNames.GDT_y.name, ]] - new_pt[0]) ** 2).sum(
                axis=1) ** 0.5 <= EPS].iloc[0]))


fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', on_click)



aaaaa
# input("输入任意键模拟新增的点")


de =  cst.interface.DesignEnvironment(mode=cst.interface.DesignEnvironment.StartMode.Existing)
def run_cst_history(cst_proj_de:cst.interface.DesignEnvironment,history_list_item:str):
    command = "\n".join(['Sub Main()',
                         history_list_item,
                         # 'RebuildOnParametricChange(False, True)',
                         'End Sub'])
    logger.info(command)
    return cst_proj_de.schematic.execute_vba_code(command)


def set_parameter(cst_proj_de:cst.interface.DesignEnvironment, parameters:dict):
    return  run_cst_history(cst_proj_de,"\n".join([*['StoreParameter("%s", %s)'%(key , parameters[key])for key in parameters],
                                                   'RebuildOnParametricChange(False, True)',]))

cst_proj_de= de.get_open_project(cst_proj_path)
t1 = time.time()
for i,pt in enumerate(new_parametric_space_pts[:]):
    t2 = time.time()
    logger.info('当前进度 = %d/%d = %.2f%%\n上次模拟耗时%.2f s'%(i,new_parametric_space_pts.shape[0],100.0*i/new_parametric_space_pts.shape[0],t2-t1))
    t1 = t2
    set_parameter(cst_proj_de,
                  {
                          ParameterNames.GDT_z.name:pt[0],
                          ParameterNames.GDT_y.name:pt[1],
                  })
    cst_proj_de.modeler.run_solver()

