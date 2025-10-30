# -*- coding: utf-8 -*-
# @Time    : 2025/10/7 0:06
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : export_Ez_map_to_CST_readable_txt.py
# @Software: PyCharm
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.interpolate
import scipy.constants  as C
import  re
def dataframe_of_TM_map_to_CST_readable_txt(df:pandas.DataFrame,field_name = "E"):
    if field_name  == "E":
        df.columns = re.split(r'\s{2,}',
                          "x [mm]           y [mm]           z [mm]       ExRe [V/m]       ExIm [V/m]       EyRe [V/m]       EyIm [V/m]       EzRe [V/m]       EzIm [V/m]", )
    elif     field_name  == "B":
        df.columns = re.split(r'\s{2,}',
                              "x [mm]           y [mm]           z [mm]       BxRe [T]       BxIm [T]       ByRe [T]       ByIm [T]       BzRe [T]       BzIm [T]", )
    else:
        raise RuntimeError("Unexpected field name ('%s')"%field_name)
    csv_path = "%s_TM.txt"%field_name
    df.to_csv(csv_path, index=False,
              #               header="""           x [mm]           y [mm]           z [mm]      x [V.s/m^2]      y [V.s/m^2]      z [V.s/m^2]
              # ------------------------------------------------------------------------------------------------------""",
              sep='\t',
              # float_format = "%.12e"
              )

    with open(csv_path, 'r') as f:
        s = f.read()
    new_s = s.replace('\t', '     ')

    with open(csv_path, 'w') as f:
        f.write(new_s)


class TM_1D_extroplator:
    def __init__(self,Ez1d_df:pandas.DataFrame,omega :float):
        """
        S. B. van der Geer和M. J. de Loos. 《General Particle Tracer User Manual (Version 3.38)》. 2008年.
        pp. 173

        ...
        Reads a 1D table of Ez samples on-axis from the specified GDF file and extrapolates these to a cylindrical symmetric field map for a TM mode cavity.
        ...

        :param Ez1d_df:
        :param omega:
        """

        self.Ez1ddata = Ez1d_df['Ez'].values
        self.zdata = Ez1d_df['z'].values
        self.pEzpz = numpy.array([*(numpy.diff(self.Ez1ddata) / numpy.diff(self.zdata)),numpy.nan])
        self.Ez1d_interpolator = scipy.interpolate.interp1d(self.zdata,self.Ez1ddata,fill_value = 0., bounds_error = False)
        self.pEzpz_interpolator = scipy.interpolate.interp1d(self.zdata,self.pEzpz,fill_value = 0., bounds_error = False)
        self.omega = omega
    def get_Ez(self,r,z):
        return self.Ez1d_interpolator(z)
    def get_Er(self,r,z):
        return -r/2* self.pEzpz_interpolator(z)

    def get_Bphi(self,r,z):
        return  - 1j*r*self.omega /(2*C.c**2 )*self.Ez1d_interpolator(z)
if __name__ == '__main__':
    Ez1d_df = pandas.read_csv(r'F:\changeworld\PyDISK\_dirty_works\SW_linac_design\GPT_rundir\Ez1d.txt',sep = r'\s+')
    f = 9.3e9
    tm1dextrapolator = TM_1D_extroplator(Ez1d_df,2*numpy.pi *f )
    plt.figure()
    zs  =numpy.arange(Ez1d_df['z'].min(),Ez1d_df['z'].max(),1e-3)
    rs = numpy.linspace(0, 2e-3, 20)
    for r in rs:
        plt.plot(zs,tm1dextrapolator.get_Ez(r,zs))

    mm  =1e-3
    plt.figure()
    for r in rs:
        plt.plot(zs,tm1dextrapolator.get_Er(r,zs),label ='r = %.2f mm'%(r/mm),)
    plt.legend()

    # R,Z = numpy.meshgrid(rs,zs,#indexing='ij'
    #                      )
    Z,R= numpy.meshgrid(zs,rs)
    plt.figure()
    cf = plt.contourf(Z,R,tm1dextrapolator.get_Er(R,Z),cmap = 'jet')
    plt.streamplot(Z,R, tm1dextrapolator.get_Ez(R,Z),tm1dextrapolator.get_Er(R,Z))
    plt.colorbar(cf)
    # plt.gca().set_aspect('equal')
    xs = numpy.linspace( -2e-3,2e-3,21)
    Z ,Y,X= numpy.meshgrid(zs,xs,xs,        indexing='ij'
    )
    R = (X ** 2 + Y ** 2) ** 0.5
    Er_tm = tm1dextrapolator.get_Er(R,Z)
    Ez_tm = tm1dextrapolator.get_Ez(R,Z)
    Ex_tm = Er_tm *  X /R
    Ey_tm = Er_tm *  Y / R

    Bphi_tm = tm1dextrapolator.get_Bphi(R, Z)
    Bx_tm = -Bphi_tm * Y / R
    By_tm = Bphi_tm * X / R


    O =  numpy.zeros(X.size)

    Edf_to_CST = pandas.DataFrame({
        "x": X.ravel() / mm,
        "y": Y.ravel() / mm,
        "z": Z.ravel() / mm,
        "Ex_real": Ex_tm.ravel(),
        "Ex_imag":O,
        "Ey_real": Ey_tm.ravel(),
        "Ey_imag": O,
        "Ez_real": Ez_tm.ravel(),
        "Ez_imag": O,
    })

    Bdf_to_CST = pandas.DataFrame({
        "x": X.ravel() / mm,
        "y": Y.ravel() / mm,
        "z": Z.ravel() / mm,
        "Bx_real": O,
        "Bx_imag":Bx_tm.imag.ravel(),
        "By_real": O,
        "By_imag": By_tm.imag.ravel(),
        "Bz_real": O,
        "Bz_imag": O,
    })
    Edf_to_CST.fillna(0.,inplace=True)
    Bdf_to_CST.fillna(0.,inplace=True)



    dataframe_of_TM_map_to_CST_readable_txt(Edf_to_CST,"E")
    dataframe_of_TM_map_to_CST_readable_txt(Bdf_to_CST,"B")