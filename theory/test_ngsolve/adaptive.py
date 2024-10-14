


import enum

# import cst.results
import numpy

import matplotlib
import pandas
import scipy
import scipy.constants as C
from scipy.fft import ifft
from scipy.optimize import curve_fit
from shapely.geometry import LineString

#from _logging import logger



matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.geom2d import SplineGeometry


#   point numbers 0, 1, ... 11
#   sub-domain numbers (1), (2), (3)
#
#
#             7-------------6
#             |             |
#             |     (2)     |
#             |             |
#      3------4-------------5------2
#      |                           |
#      |             11            |
#      |           /   \           |
#      |         10 (3) 9          |
#      |           \   /     (1)   |
#      |             8             |
#      |                           |
#      0---------------------------1
#

def MakeGeometry():
    geometry = SplineGeometry()

    # point coordinates ...
    pnts = [ (0,0), (1,0), (1,0.6), (0,0.6), \
             (0.2,0.6), (0.8,0.6), (0.8,0.8), (0.2,0.8), \
             (0.5,0.15), (0.65,0.3), (0.5,0.45), (0.35,0.3) ]
    # 点的编号
    pnums = [geometry.AppendPoint(*p) for p in pnts]

    # start-point, end-point, boundary-condition, domain on left side, domain on right side:
    # domain 0 表示外部区域（不计算）
    lines = [ (0,1,1,1,0), (1,2,2,1,0), (2,5,2,1,0), (5,4,2,1,2), (4,3,2,1,0), (3,0,2,1,0), \
              (5,6,2,2,0), (6,7,2,2,0), (7,4,2,2,0), \
              (8,9,2,3,1), (9,10,2,3,1), (10,11,2,3,1), (11,8,2,3,1) ]

    for p1,p2,bc,left,right in lines:
        geometry.Append( ["line", pnums[p1], pnums[p2]], bc=bc, leftdomain=left, rightdomain=right)
    return geometry



mesh = Mesh(MakeGeometry().GenerateMesh (maxh=0.2))


# HP有限元
"""
有限元方法按照收敛格式的不同主要可分为三类：  

第一类有限元方法称为h-型(h-version)有限元，其中h代表单元的最大尺寸。在计算过程中，该方法不改变单元的类型而是通过不断缩小单元的几何尺寸，即加密网格的方式来改进计算结果。
由于该方法中单元的阶次一般较低，因而又称为低阶有限元方法。  

第二类有限元方法称为p-型(p-version)有限元方法，其中p为计算中使用的单元的最低阶次。在计算过程中，该方法不改变结构的网格划分而是通过提高单元的最低阶次来改进计算结果。
由于该方法中一般使用的单元阶次较高，因而被称为高阶有限元方法。  

第三类有限元方法则是这两种的结合形式，称为hp-型(hp-version)有限元方法。该方法在计算过程中往往同时调整结构计算的网格尺寸以及单元的阶次来提高计算结果的精度。参考文献[1]刘波, 伍洋, 邢誉峰. 微分求积升阶谱有限元方法[M]. 国防工业出版社, 2019.

作者：7hefoo1
链接：https://www.zhihu.com/question/328003141/answer/2254703402
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""
fes = H1(mesh, order=3, dirichlet=[1], autoupdate=True)
u = fes.TrialFunction()
v = fes.TestFunction()

# one heat conductivity coefficient per sub-domain
lam = CoefficientFunction([1, 1000, 10])
a = BilinearForm(fes, symmetric=False)
a += lam*grad(u)*grad(v)*dx


# heat-source in sub-domain 3
f = LinearForm(fes)
f += CoefficientFunction([0, 0, 1])*v*dx

c = MultiGridPreconditioner(a, inverse = "sparsecholesky")

gfu = GridFunction(fes, autoupdate=True)
Draw (gfu)

# finite element space and gridfunction to represent
# the heatflux:
space_flux = HDiv(mesh, order=2, autoupdate=True)
gf_flux = GridFunction(space_flux, "flux", autoupdate=True)

def SolveBVP():
    a.Assemble()
    f.Assemble()
    inv = CGSolver(a.mat, c.mat)
    gfu.vec.data = inv * f.vec
    Redraw (blocking=True)



l = []

def CalcError():
    flux = lam * grad(gfu)
    # interpolate finite element flux into H(div) space:
    gf_flux.Set (flux)

    # Gradient-recovery error estimator
    err = 1/lam*(flux-gf_flux)*(flux-gf_flux)
    elerr = Integrate (err, mesh, VOL, element_wise=True)

    maxerr = max(elerr)
    l.append ( (fes.ndof, sqrt(sum(elerr)) ))
    print ("maxerr = ", maxerr)

    for el in mesh.Elements():
        mesh.SetRefinementFlag(el, elerr[el.nr] > 0.25*maxerr)

with TaskManager():
    while fes.ndof < 10000:
        SolveBVP()
        CalcError()
        mesh.Refine()

SolveBVP()


import matplotlib.pyplot as plt
plt.ion()

plt.yscale('log')
plt.xscale('log')
plt.xlabel("ndof")
plt.ylabel("H1 error-estimate")
ndof,err = zip(*l)
plt.plot(ndof,err, "-*")

# plt.show()

input("<press enter to quit>")


pts = [v.point for v in mesh.vertices]
from scipy.spatial import Delaunay
# tri = Delaunay(pts)
plt.figure()
plt.triplot(*zip(*pts), [[mesh.faces[i].vertices[j].nr for j in range(3)] for i in range(mesh.nface)])


