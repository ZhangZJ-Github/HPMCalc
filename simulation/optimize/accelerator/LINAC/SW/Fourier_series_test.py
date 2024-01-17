# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 0:46
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : Fourier_series_test.py
# @Software: PyCharm

import numpy
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('tkagg')
import  matplotlib.pyplot as plt
plt.ion()
np = numpy
def _Fourier_series(t, T0, a,b):
    """
    傅里叶级数
    :param t:
    :return:
    """
    ret = a[0]
    for n in range(1, len(a)):
        phin = n * 2*np.pi / T0 * t
        ret +=( a[n] * np.cos(phin) + b[n]*np.sin(phin))
    return ret

L0 = 1
def f1(z):
    # L0= 1
    # return    numpy.cos(2*numpy.pi *(1/L0+0*0.07*z)*z)+2
    return 2*(z+5)**2
zs = numpy.linspace(-5,5, 2000)
f1s = f1(zs)
plt.figure()
plt.plot(zs, f1s,label = 'original')
plt.legend()


L0 = 10
params , cov = curve_fit(lambda z, *A:
                     _Fourier_series(z, L0, *(numpy.array(A).reshape(2, -1))), zs, f1s, p0=numpy.array([1.] * 30))
As,Bs  =  params.reshape(2, -1)
plt.plot(zs, _Fourier_series(zs, L0, As,Bs), label ='Fourier series')

fig,axs = plt.subplots(2,1,sharex=True, sharey=True)
axs[0].bar(range(len(As)),As,label = 'A')
axs[1].bar(range(len(Bs)),Bs,label = 'B')
(ax.legend() for ax  in  axs)
