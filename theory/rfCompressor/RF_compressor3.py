# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 17:02
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : RF_comporessor3.py
# @Software: PyCharm
import numpy, matplotlib,pandas,scipy
import cst.results
matplotlib.use('tkagg')
import matplotlib .pyplot as plt
from scipy.optimize import curve_fit
plt.ion()
# omega = numpy.linspace(8.0e9,10e9,100)
# Q0 = 20
# omega0 = 9.e9
# Chi = Q0*(omega/omega0-omega0/omega0)
# plt.plot(omega, numpy.abs(1/ (1+Chi**2)**0.5))
f_target = 9.3e9


key_complex = 'complex'
key_period_for_calculate_avg = 'period_for_calculate_avg'
key_square = 'square'
def to_df(o23):
    df_o23 = pandas.DataFrame(o23)
    df_o23[key_period_for_calculate_avg] = df_o23[0] * 1e-9 // (5 / f_target)
    df_o23[key_square] = df_o23[1] ** 2
    df_time_avg = df_o23.groupby(key_period_for_calculate_avg).mean()
    df_o23['interpolated_periodic_avg_square'] = numpy.interp(df_o23[0], df_time_avg[0], df_time_avg['square'],)
    df_o23[key_complex] = scipy.signal.hilbert(df_o23[1])
    return df_o23
class TwoPortsNetwork:
    def __init__(self,QL,omega0,C,):
        self.QL = QL
        self.omega0 = omega0
        self.C = C
    def __f(self,omega):
        return self.omega0**2 - omega**2 +1j*self.omega0*omega/self.QL
    def S21(self, omega,n1,Z1,n2,Z2):
        return 1j*omega/(n1*n2*(Z1*Z2)**0.5  * self.C *self.__f(omega))
    def S11(self,omega,n1, Z1):
        """
        port 1表示测试信号的输入端口
        :param omega:
        :param n1:
        :param Z1:
        :return:
        """
        return 1j*omega/(n1**2*Z1*self.C*self.__f(omega))-1
proj: cst.results.ProjectFile = cst.results.ProjectFile(r"E:\CSTprojects\GeneratorAccelerator\test_SES_switch01.1.cst",
                                                        allow_interactive=True)
# o23 = numpy.array(proj_discharging.get_3d().get_result_item(r'1D Results\Port signals\o2,3[1.0,0.0,signal1],[f_target]').get_data())
# o33 = numpy.array(proj_discharging.get_3d().get_result_item(r'1D Results\Port signals\o3,3[1.0,0.0,signal1],[f_target]').get_data())
# i3 = numpy.array(proj_discharging.get_3d().get_result_item(r'1D Results\Port signals\i3[1.0,0.0,signal1],[9.3]').get_data())

S33 =numpy.array(proj.get_3d().get_result_item('1D Results\\S-Parameters\\S3,3').get_data())
S23 =numpy.array(proj.get_3d().get_result_item('1D Results\\S-Parameters\\S2,3').get_data())
Zref_2 = numpy.array(proj.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 2(1)').get_data())
Zref_3 = numpy.array(proj.get_3d().get_result_item('1D Results\\Reference Impedance\\ZRef 3(1)').get_data())

Zref_3_interpolator = scipy.interpolate.interp1d(numpy.real(Zref_3[:,0]),numpy.real(Zref_3[:, 1]),fill_value="extrapolate")
Zref_2_interpolator = scipy.interpolate.interp1d(numpy.real(Zref_2[:,0]),numpy.real(Zref_2[:, 1]),fill_value="extrapolate")
_func_S11 = lambda omega, omega0, QL, Cn12:numpy.abs(TwoPortsNetwork(QL, omega0,Cn12).S11(omega,1,Zref_3_interpolator(omega/(2*numpy.pi))))

trusted_index =  (numpy.abs(S33[:, 0])>9.26)&(numpy.abs(S33[:, 0])<9.35)
(omega0, QL, Cn12 ), cov = curve_fit(_func_S11, numpy.abs(S33[:, 0][trusted_index]) *(2*numpy.pi),numpy.abs(S33[:, 1][trusted_index]),p0=[2*numpy.pi*9.3, 100,0.2])
f0 = omega0 / (2*numpy.pi)
plt.figure()
plt.plot(numpy.abs(S33[:, 0]) ,numpy.abs(S33[:, 1]),label = 'S33, original')
# plt.plot(numpy.abs(S33[:, 0]) , _func_S11(numpy.abs(S33[:, 0]) *(2*numpy.pi),omega0, QL, Cn12 ),label = 'S33, equivalent circuit')
fs = numpy.linspace(8,10,2000)
plt.plot(fs , _func_S11(fs *(2*numpy.pi),omega0, QL, Cn12 ),label = 'S33, equivalent circuit')
plt.legend()

