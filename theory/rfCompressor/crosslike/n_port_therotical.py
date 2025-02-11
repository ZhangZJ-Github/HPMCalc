# -*- coding: utf-8 -*-
# @Time    : 2025/1/14 15:19
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : n_port_therotical.py
# @Software: PyCharm
import numpy
from  sympy.matrices import Matrix,eye,diag,zeros
import sympy
from _logging import logger
N_ports = 6
indexes_mismatched_port = [1,4,5] # index从0开始
index_of_true_inputing_port = 0
a_in = sympy.symbols('a_in',zero = False)#1+0j
# 变量名下标从1开始
Gamma_of_mismatched_port =Matrix( sympy.symbols("".join(["\Gamma_{%d} "%(i+1) for i in indexes_mismatched_port]) ,real = False))
S = Matrix([sympy.symbols("".join(["S_{{%d}{%d}} "%(i+1,j+1) for j in range(N_ports)]) ,real = False) for i in range(N_ports)])

# 假设互易，减少变量数
for i in range(N_ports):
    for j in range(i+1):
        S[i,j] = S[j,i]
port_pairs = {0:1,4:5,2:3}#[[0,1],[4,5],[2,3]]
changged = numpy.zeros(S.shape,bool)
for i in range(N_ports):
    eq_port_i = port_pairs.get(i, None)
    if  eq_port_i is None:
        continue
    for j in range(N_ports):
        eq_port_j = port_pairs.get(j,None)
        if eq_port_j is None:continue
                # and  not changged[eq_port_i,eq_port_j]:
        logger.info("%s = %s"%(S[eq_port_i,eq_port_j],S[i,j]))
        S[eq_port_i,eq_port_j] = S[i,j]
        changged[eq_port_i,eq_port_j] = True






Gamma_of_mismatched_port[2] = Gamma_of_mismatched_port[1]

# rows_mismatched_port,cols_mismatched_port = numpy.meshgrid(indexes_mismatched_port,indexes_mismatched_port,indexing= 'ij')

S_mismatched = S[indexes_mismatched_port,indexes_mismatched_port]
M = eye(S_mismatched.shape[0])  - S_mismatched*diag(*Gamma_of_mismatched_port)
b_mismatch = (M)**-1 * S[indexes_mismatched_port,index_of_true_inputing_port]*a_in
a = zeros(N_ports,1)
a[index_of_true_inputing_port] =  a_in

_temp1 = Gamma_of_mismatched_port.multiply_elementwise(b_mismatch)
for i,ii in enumerate(indexes_mismatched_port):
    a[ii,0] = _temp1[i]
b = S*a
# sympy.init_printing(use_latex = 'svg')
to_be_printed = (b[2,0]/ a_in).simplify()
sympy.print_latex(to_be_printed)

logger.info(sympy.latex(to_be_printed,#long_frac_ratio=2
                        ))
# sol = sympy.solve(to_be_printed, Gamma_of_mismatched_port[1])
