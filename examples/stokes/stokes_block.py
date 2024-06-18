from ngsolve import *
# from ngsolve.webgui import Draw
import netgen.geom2d as g2d
import netgen.csg as csg
from ngsolve.krylovspace import *

# such that we find the "amg_utils.py" from the elasticity folder
import sys
sys.path.append("../elasticity")

from amg_utils import *
from stokes_utils import *
from tesla import *

from NgsAMG import *

order = 2
maxh  = 0.1
dim  = 2
nu = 1e-3

if False:
    # N controls the number of valves (one valve consisting of an upper and lower loop)
    valve = GetValve(N = 1, dim = dim, R = 0.5, alpha = 25, Winlet = 1, beta = 180, L1 = 6.4, L2 = 7, Linlet = 5, closevalve = True)
    mesh = Mesh(OCCGeometry(valve, dim=dim).GenerateMesh(maxh = 0.5))
    mesh.Curve(3)
    diri="wall|valve"
    uin = None
    f_vol = CF((1,0))
else:
    uin, geo = geo_2dchannel(L=30)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    diri="wall|inlet"
    f_vol = None


# Draw(mesh)

# (u, uhat), p
V, Q, a, b, aPen = StokesHDGDiscretization(mesh, order=order, diri=diri, nu=nu)

aInv = Preconditioner(a, "direct")

p_mass = BilinearForm(Q)
p,q = Q.TnT()
p_mass += 1.0/nu * p * q * dx

schur_pre = Preconditioner(p_mass, "local")

a.Assemble()
b.Assemble()
p_mass.Assemble()

(u, uhat), (v, vhat) = V.TnT()
if f_vol is not None:
    f = LinearForm( f_vol * v * dx(definedon = mesh.Materials("valve"))).Assemble() 
else:
    f = LinearForm(V).Assemble()

g = LinearForm(Q).Assemble()

m = BlockMatrix([[a.mat, b.mat.T], [b.mat, None]])
m_pre = BlockMatrix([[aInv, None], [None, schur_pre]])

rhs = BlockVector([f.vec, g.vec])

if uin is not None:
    gf_in = GridFunction(V)
    gf_in.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
    f.vec.data -= a.mat * gf_in.vec

gfu = GridFunction(V)
gfp = GridFunction(Q)
sol = BlockVector([gfu.vec, gfp.vec])

gmr = GMResSolver(mat=m, pre=m_pre, maxiter=500, tol=1e-8, printrates=True)

gmr.Solve(rhs=rhs, sol=sol)

if uin is not None:
    gfu.components[0].vec.data += gf_in.components[0].vec

# print(gfu.vec)

# print("LAMS = ", lams)

