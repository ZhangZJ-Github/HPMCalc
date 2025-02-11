# 2024年12月23日19:43:49
import matplotlib.pyplot as plt
import numpy
from simulation.post_processing .GPT_trajectory import GPTTraj
import pygpt

import enum
import typing

import cst.results
import matplotlib
import numpy
import pandas
import scipy
import skrf
import os
from scipy.fft import ifft
from skrf.network import Network

from simulation.task_manager.simulator import  df_to_gdf

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.constants as C

from _logging import logger
plt.ion()
color_table = list(matplotlib.colors.TABLEAU_COLORS.keys())

gdf =  pygpt.gdftomemory(r"E:\CSTprojects\gMOT_CAES\pp\GPT\test_acc.gdf")
traj_gdf=  pygpt.gdftomemory(r"E:\CSTprojects\gMOT_CAES\pp\GPT\traj.gdf")

z_grating = 0
z_atom_cloud_center = -9.05e-3

N_pts = len(traj_gdf)
plt.figure(figsize=(4,3), constrained_layout = True)
for i in numpy.array(range(N_pts)) [numpy.random.rand(N_pts)> 0.97]:
    plt.scatter(traj_gdf[i]['d']['z'] [0]/ 1e-3,traj_gdf[i]['d']['x'][0] / 1e-6, alpha = 0.5,c = color_table[1],#s= 1
                )
    plt.plot(traj_gdf[i]['d']['z'] / 1e-3,traj_gdf[i]['d']['x'] / 1e-6, alpha = 0.5)
plt.xlabel('z (mm)')
plt.ylabel('x ($\mu m$)')
plt.axvline(z_grating, ls = ":")
plt.xlim(-8, 2)
plt.ylim(-800,800)


gpttraj = GPTTraj(traj_gdf)
par_ids = gpttraj.dfs.keys()
screens = numpy.linspace(-4e-3 , 2e-3, 10)
sigma_x = gpttraj.std_at_screen(-4.0e-3, "x")
data_at_out_screen = gpttraj.interpolate_at_screen(2e-3,)



plt.figure(figsize=(4,3), constrained_layout = True)
plt.scatter(data_at_out_screen['x'] / 1e-6,data_at_out_screen['Bx'] / data_at_out_screen['Bz'],s = 0.01)
plt.xlabel('x ($\mu m$)')
plt.ylabel('x\'')

plt.figure(figsize=(4,3), constrained_layout = True)
plt.scatter(data_at_out_screen['x'] / 1e-6,data_at_out_screen['y'] / 1e-6,s = 0.1)
plt.gca().set_aspect('equal')
plt.xlabel('x ($\mu m$)')
plt.ylabel('y ($\mu m$)')


std_gdf =  pygpt.gdftomemory(r"E:\CSTprojects\gMOT_CAES\pp\GPT\std.gdf")
plt.figure(figsize=(4,3), constrained_layout = True)
plt.plot(std_gdf[0]['d']['position']/1e-3,std_gdf[0]['d']['nemixrms'] /1e-9)
plt.axvline(z_grating, ls = ":")
# plt.xlim(-8, 2)
plt.xlabel('z (mm)')
plt.ylabel('$\epsilon_{n,x}$ (nm rad)')


plt.figure(figsize=(4,3), constrained_layout = True)
plt.plot(std_gdf[0]['d']['position']/1e-3, (std_gdf[0]['d']['avgG'] - 1)*C.m_e * C.c**2 /( C.eV)  /1e3)
plt.xlabel('z (mm)')
plt.axvline(z_grating, ls = ":")
plt.ylabel('kinetic energy (keV)')
plt.xlim(-8, 2)
