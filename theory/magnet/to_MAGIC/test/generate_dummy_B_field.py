# -*- coding: utf-8 -*-
# @Time    : 2025/10/23 0:46
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : generate_dummy_B_field.py
# @Software: PyCharm
import pandas

import numpy
rs,zs = numpy.linspace(0, 5e-3,10,),numpy.linspace(740e-3,1000e-3,1000)
R,Z = numpy.meshgrid(rs,zs,indexing= 'ij')
def get_B(r,z,L):
    return (0+ 1*numpy.sin(z / L * 2*numpy.pi)) * 0.4
B = get_B(R,Z,15e-3)
Gs =1e-4
mm = 1e-3
O = B.ravel() * 0 / Gs
df = pandas.DataFrame({
    "R":R.ravel()/mm,
    "Z":Z.ravel() / mm,
    "Br":O,
    "Bz":B.ravel()/Gs,
    "|B|":O,
    "A":O,
    "dBz/dr":O,
    "dBr/dr":O,
    "F":O,

})
header = """ Magnetic fields for a rectangular area with corners at:
 (Rmin, Zmin) = (   %.4E,  %.4E)
 (Rmax, Zmax) = (   %.4E,  %.4E)
 R and Z increments:    %d   %d
 
       R         Z          Br            Bz            |B|          A           dBz/dr        dBr/dz         F
"""%(*(numpy.array([R.min(),Z.min(),R.max(),Z.max()]) / mm),
     len(rs),len(zs))
from io import StringIO

sio = StringIO()
temp_csv_path = "temp.csv"
# df.columns  = None
df.to_csv(temp_csv_path,sep = '\t',index=False,
    header = None,
          #header = header,
          # float_format = "%13.6e",
          float_format = "%.3e",
          )

with open(temp_csv_path, 'r') as f:
    csv_lines =  f.readlines()
# unit = "     (mm) (mm)         (G)           (G)          "
unit = "     (mm)      (mm)         (G)           (G)           (G)         (G-cm)       (G/cm)        (G/cm)         Index"
csv_lines.insert(0, unit+"\n")#[ csv_lines[0]] + [unit] + csv_lines[1:]

text = header + ''.join(csv_lines)
# text = text.replace("\t"," ")
with open("OUTSF7.TXT",'w') as f:
    f.write(text)
