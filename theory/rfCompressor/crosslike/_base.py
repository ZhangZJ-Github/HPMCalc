# -*- coding: utf-8 -*-
# @Time    : 2025/1/17 18:23
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _base.py
# @Software: PyCharm

import time
import typing

import cst.results
import cst.results
import matplotlib

matplotlib.use('tkagg')
import skrf
from skrf.media.rectangularWaveguide import RectangularWaveguide

from _logging import logger

import numpy
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

plt.ion()

key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
key_interpolated_periodic_avg_square = 'interpolated_periodic_avg_square'
f_target = 9.3e9
f_target_GHz = f_target / 1e9


def get_interpolator_on_dataset(cst_proj_path: str, indexing_parameter_names: typing.List[str]):
    t1 = time.time()
    proj_3D: cst.results.ProjectFile = cst.results.ProjectFile(
        cst_proj_path,
        allow_interactive=True)

    # S = {}

    # key_data = 'data'
    # key_interpolator = "interpolator"
    # key_run_id = "runid"
    run_ids: list = proj_3D.get_3d().get_all_run_ids()
    try:
        run_ids.remove(0)
    except ValueError as e:
        logger.info("run_id = 0 is not included in the dataset.\n%s" % e)

    data_set = {}
    for run_id in run_ids:
        # run_id = 0
        parameter_combination = proj_3D.get_3d().get_parameter_combination(run_id)
        indexing_parameters = [parameter_combination[key] for key in indexing_parameter_names]
        tree_items = proj_3D.get_3d().get_tree_items()
        S_param_items = []

        for tree_item in tree_items:
            if tree_item.startswith('1D Results\\S-Parameters\\'):
                S_param_items.append(tree_item)
        s_param_example = numpy.array(
            proj_3D.get_3d().get_result_item(S_param_items[0],
                                             run_id).get_data())
        Nf = len(s_param_example)
        N_ports = int(len(S_param_items) ** 0.5)
        z0 = numpy.zeros((Nf, N_ports), )
        s = numpy.zeros((Nf, N_ports, N_ports), complex)
        f = s_param_example[:, 0].real
        for i in range(N_ports):
            # TODO: for debug
            # if i % 3 > 1: continue

            for j in range(N_ports):
                # if j%3 >1:continue
                item = numpy.array(
                    proj_3D.get_3d().get_result_item(S_param_items[i * N_ports + j],
                                                     run_id).get_data())
                s[:, i, j] = item[:, 1]
                if i == j:
                    z0[:, i] = item[:, 2].real
        data_set[run_id] = {
            "parameter_combination": parameter_combination,
            "indexing_parameters": indexing_parameters,
            "f": f,
            "s": s,
            "z0": z0
        }
    _indexing_parameters = numpy.array([data_set[run_id]["indexing_parameters"] for run_id in data_set.keys()])
    kwargs_for_interpolator = {}
    if len(indexing_parameter_names) == 1:
        Interpolator = interp1d
        _indexing_parameters = _indexing_parameters[:, 0]
        kwargs_for_interpolator["axis"] = 0
    else:
        Interpolator = LinearNDInterpolator

    s_interpolator_on_dataset = Interpolator(_indexing_parameters,
                                             numpy.array([data_set[run_id]["s"] for run_id in data_set.keys()]),
                                             **kwargs_for_interpolator
                                             )
    z0_interpolator_on_dataset = Interpolator(_indexing_parameters,
                                              numpy.array([data_set[run_id]["z0"] for run_id in data_set.keys()]),
                                              **kwargs_for_interpolator
                                              )
    logger.info("Time cost for build dataset and interpolator = %.4f s" % (time.time() - t1))
    return s_interpolator_on_dataset, z0_interpolator_on_dataset, data_set


def get_interpolator(E_data):
    return interp1d(E_data[:, 0].real, E_data[:, 1])


def calculate_signal_outgoing_mismatched_ports(  # nw: skrf.Network,
        S_matrix,
        # f,
        indexes_of_mismatched_ports,
        reflection_coefficients_of_mismatched_ports: numpy.ndarray,
        index_of_port_to_input_signal: int = 0,
        a_input_port=1 + 0j):
    """

    :param nw:
    :param f:
    :param indexes_of_mismatched_ports: shape (n,)
    :param reflection_coefficients_of_mismatched_ports: shape (n,), dtype=complex
    :return:
    """
    # indexes_of_mismatched_ports = numpy.array(indexes_of_mismatched_ports)
    # interpolater_nw = nw.interpolate([f])
    # bool_index = numpy.zeros(interpolater_nw.s.shape[1],dtype=bool)
    # bool_index[indexes_of_mismatched_ports] =True
    rows, columns = numpy.meshgrid(indexes_of_mismatched_ports, indexes_of_mismatched_ports, indexing='ij')
    # s_all_ports = s_matrix#nw.interpolate([f]).s[0]
    s = S_matrix[rows, columns]
    Gamma = numpy.matrix(numpy.diag(reflection_coefficients_of_mismatched_ports,
                                    ))  # .astype(complex)

    M = numpy.matrix(numpy.eye(*s.shape, )) - numpy.matrix(s) * Gamma
    si = numpy.matrix(S_matrix[indexes_of_mismatched_ports, index_of_port_to_input_signal]).T

    return numpy.linalg.inv(M) * si * a_input_port


def calculate_signals_outgoing(  # nw: skrf.Network,
        S_matrix,
        # f,
        indexes_of_mismatched_ports,
        reflection_coefficients_of_mismatched_ports: numpy.ndarray,
        index_of_port_to_input_signal: int = 0,
        a_input_port=1 + 0j):
    as_ = numpy.matrix(numpy.zeros(  # nw.nports,
        S_matrix.shape[0],
        dtype=complex)).T
    as_[index_of_port_to_input_signal] = a_input_port

    signal_outgoing_mismatched_ports = calculate_signal_outgoing_mismatched_ports(S_matrix, indexes_of_mismatched_ports,
                                                                                  reflection_coefficients_of_mismatched_ports,
                                                                                  index_of_port_to_input_signal).A.ravel()
    as_[indexes_of_mismatched_ports] = numpy.matrix(
        reflection_coefficients_of_mismatched_ports * signal_outgoing_mismatched_ports).T
    return numpy.matrix(S_matrix) * as_


class CrossLikeNetwork:
    def __init__(self, nw_cross: skrf.Network,
                 map_from_port_id_to_matched_waveguide: typing.Dict[int, RectangularWaveguide]):
        self.nw_cross = nw_cross
        self.map_from_port_id_to_matched_waveguide = map_from_port_id_to_matched_waveguide

    def update_network(self, new_network: skrf.Network):
        self.nw_cross = new_network

    def get_nw_connected_with_delay_shorts(
            self, map_from_port_id_to_beta_l_delay_short: typing.Dict[int, float],
            # port_id_connect_to_delayshorts: typing.List[int],
            # rect_wGs: typing.List[RectangularWaveguide],
            # beta_l_delayshorts: typing.List[float],
            # port_id_outs,
            # port_ids_to_monitor_Emax =None,
    ) -> typing.List[skrf.Network]:
        # if port_ids_to_monitor_Emax == None:port_ids_to_monitor_Emax = port_id_connect_to_delayshorts
        list_nw_cross = [self.nw_cross]
        port_ids_connect_to_delayshorts_in_newsest_nw = list(map_from_port_id_to_beta_l_delay_short.keys())
        for i, each_port_id_connect_to_delayshorts_in_initial_nw in enumerate(
                map_from_port_id_to_beta_l_delay_short):
            nw_cross_1 = skrf.connect(
                list_nw_cross[-1], port_ids_connect_to_delayshorts_in_newsest_nw[0],
                self.map_from_port_id_to_matched_waveguide[
                    each_port_id_connect_to_delayshorts_in_initial_nw].delay_short(
                    map_from_port_id_to_beta_l_delay_short[each_port_id_connect_to_delayshorts_in_initial_nw],
                    unit='rad'), 0)
            _connected_id = port_ids_connect_to_delayshorts_in_newsest_nw.pop(0)
            port_ids_connect_to_delayshorts_in_newsest_nw = [port_id - (1 if _connected_id < port_id else 0) for port_id
                                                             in
                                                             port_ids_connect_to_delayshorts_in_newsest_nw]
            list_nw_cross.append(nw_cross_1)
        return list_nw_cross

    @staticmethod
    def get_port_ids_in_new_nw(port_ids_in_old_network, port_ids_no_longer_exist) -> typing.List[int]:
        assert (len(set(port_ids_no_longer_exist).union(set(port_ids_in_old_network))) == len(
            port_ids_no_longer_exist) + len(port_ids_in_old_network)), "两个列表中不应有重复元素"
        N_smaller_id_in_port_id_connect_to_delayshorts = []
        for port_id_out in port_ids_in_old_network:
            _n = 0
            for port_id_connect_to_delayshort in port_ids_no_longer_exist:
                if port_id_connect_to_delayshort < port_id_out: _n += 1
            N_smaller_id_in_port_id_connect_to_delayshorts.append(_n)
        port_id_outs_in_newest_nw = numpy.array(port_ids_in_old_network) - numpy.array(
            N_smaller_id_in_port_id_connect_to_delayshorts)
        return port_id_outs_in_newest_nw

    def get_out_S_parameters(self, map_from_port_id_to_beta_l_delay_short: typing.Dict[int, float],
                             # port_id_connect_to_delayshorts: typing.List[int],
                             # rect_wGs: typing.List[RectangularWaveguide],
                             # beta_l_delayshorts: typing.List[float],
                             port_id_outs: typing.List[int], port_id_input=0, f_target=f_target):
        nws = self.get_nw_connected_with_delay_shorts(map_from_port_id_to_beta_l_delay_short
                                                      # dict(zip(port_id_connect_to_delayshorts, beta_l_delayshorts))
                                                      )
        port_id_outs_in_newest_nw = self.get_port_ids_in_new_nw(port_id_outs,
                                                                list(map_from_port_id_to_beta_l_delay_short.keys()))
        return nws[-1].interpolate([f_target]).s[0, port_id_outs_in_newest_nw, port_id_input]

    def get_total_out_power_ratio(self, map_from_port_id_to_beta_l_delay_short,
                                  # nw_cross, port_id_connect_to_delayshorts: typing.List[int],
                                  #                       rect_wGs: typing.List[RectangularWaveguide],
                                  #                       beta_l_delayshorts: typing.List[float],
                                  port_id_outs: typing.List[int], port_id_input=0, f_target=f_target):
        S_param_outs = self.get_out_S_parameters(
            # nw_cross, port_id_connect_to_delayshorts,
            #                                 rect_wGs, beta_l_delayshorts,
            map_from_port_id_to_beta_l_delay_short, port_id_outs,
            port_id_input, f_target)
        return numpy.sum(numpy.abs(S_param_outs) ** 2, axis=-1)
