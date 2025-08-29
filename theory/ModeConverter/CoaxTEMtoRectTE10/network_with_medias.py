# -*- coding: utf-8 -*-
# @Time    : 2025/7/20 21:50
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : network_with_medias.py
# @Software: PyCharm

import typing

import numpy
import skrf.frequency
from skrf.media.media import DefinedGammaZ0
import cst.results
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
class NetWorkwithMedias:
    @staticmethod
    def get_map_port_name_in_CST_to_index(map_index_to_port_name_in_CST):
        return {map_index_to_port_name_in_CST[k]:k for k in map_index_to_port_name_in_CST}

    def __init__(self, nw:skrf.Network, medias:typing.List[skrf.media.Media],
                 map_index_to_port_name_in_CST:typing.Dict[int,str],map_port_name_in_CST_to_index:typing.Dict[str,int] =None):
        self.nw= nw
        self.medias =medias
        self.map_index_to_port_name_in_CST =map_index_to_port_name_in_CST
        self.map_port_name_in_CST_to_index=map_port_name_in_CST_to_index
        pass
    @staticmethod
    def from_CST(#self,
                 cst_proj_path,run_id = 0):
        proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
            cst_proj_path,
        allow_interactive=True)

        item_names = proj_3D.get_3d().get_tree_items()
        Zref_names = []
        S_param_names = []
        Gamma_names = []
        for item_name in item_names:
            if item_name.startswith("1D Results\\Reference Impedance\\ZRef"):
                Zref_names.append(item_name)
            if item_name.startswith("1D Results\\Port Information\\Gamma" ) :
                Gamma_names.append(item_name)
            if item_name.startswith("1D Results\\S-Parameters\\S"):
                S_param_names.append(item_name)

        example_data =numpy.array(proj_3D.get_3d().get_result_item(Gamma_names[0],run_id).get_data())
        S_param_example_data =numpy.array(proj_3D.get_3d().get_result_item(S_param_names[0],run_id).get_data())
        medias = []
        S_params = numpy.zeros((S_param_example_data.shape[0],len(Gamma_names),len(Gamma_names),),dtype=complex)
        len1 = len('1D Results\\Port Information\\Gamma\\')
        map_index_to_port_name_in_CST = {i: gamma_name [len1:]for i, gamma_name in enumerate(Gamma_names)
        }
        map_port_name_in_CST_to_index = NetWorkwithMedias.get_map_port_name_in_CST_to_index(map_index_to_port_name_in_CST)#{map_index_to_port_name_in_CST[k]:k for k in map_index_to_port_name_in_CST}
        frequency_unit = 1e9
        for i,item_name in enumerate(Gamma_names):
            gamma_data = numpy.array(proj_3D.get_3d().get_result_item(item_name,run_id).get_data())
            medias.append((
                DefinedGammaZ0(frequency= skrf.frequency.Frequency.from_f(gamma_data[:,0].real * frequency_unit),
                              gamma = numpy.array(proj_3D.get_3d().get_result_item(item_name,run_id).get_data())[:, 1] ,
                              )#,item_name[len1:]
            )
                           )
        len2 = len("1D Results\\S-Parameters\\S")
        for item_name in S_param_names:
            port_name_1 ,port_name_2 = port_names_in_CST = item_name[len2:].split(",")
            S_params[:,map_port_name_in_CST_to_index[port_name_1],map_port_name_in_CST_to_index[port_name_2]] = numpy.array(proj_3D.get_3d().get_result_item(item_name,run_id).get_data())[:, 1]
            pass
        nw = skrf.Network(frequency=skrf.frequency.Frequency.from_f(S_param_example_data[:,0].real * frequency_unit),s = S_params)
        if 0:
            plt.figure()
            nw.plot_s_mag()

        return NetWorkwithMedias(nw, medias, map_index_to_port_name_in_CST, map_port_name_in_CST_to_index)