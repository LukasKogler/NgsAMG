from ngsolve import *
# from ngsolve.webgui import Draw
import netgen.geom2d as g2d
import netgen.csg as csg
from ngsolve.krylovspace import *

from NgsAMG import *

order = 3
maxh  = 0.1

def geo_2dchannel(H=0.41, L=2.2, obstacle=True):
    geo = g2d.SplineGeometry()
    geo.AddRectangle((0, 0), (L,H), bcs=("wall", "outlet", "wall", "inlet"))
    uin = CoefficientFunction( (1.5 * (2/H)**2 * y * (H - y), 0))
    if obstacle == True:
        pos = 0.2
        r = 0.05
        geo.AddCircle((pos, pos), r=r, leftdomain=0, rightdomain=1, bc="cyl")
    return uin, geo

def geo_3dchannel(H=0.41, L=0.41, W=2.2, obstacle=True):
    geo = csg.CSGeometry()
    channel = csg.OrthoBrick( csg.Pnt(-1, 0, 0), csg.Pnt(L+1, H, W) ).bc("wall")
    inlet = csg.Plane (csg.Pnt(0,0,0), csg.Vec(-1,0,0)).bc("inlet")
    outlet = csg.Plane (csg.Pnt(L, 0,0), csg.Vec(1,0,0)).bc("outlet")
    uin = CoefficientFunction( (2.25 * (2/H)**2 * (2/W)**2 * y * (H - y) * z * (W - z), 0, 0))
    if obstacle == True:
        pos = (0.5, 0.2)
        r = 0.05
        cyl = csg.Cylinder(csg.Pnt(pos[0], pos[1], 0), csg.Pnt(pos[0], pos[1], 1), r).bc("obstacle")
        fluiddom = channel*inlet*outlet-cyl
    else:
        fluiddom = channel*inlet*outlet
    geo.Add(fluiddom)
    return uin, geo

# uin, geo = geo_2dchannel()
uin, geo = geo_3dchannel()
mesh = Mesh(geo.GenerateMesh(maxh=maxh))

V1 = HDiv ( mesh, order = order, dirichlet = "wall|cyl|inlet", hodivfree=True )
V2 = TangentialFacetFESpace(mesh, order = order, dirichlet = "wall|cyl|inlet" )
Q = L2( mesh, order = 0)#order-1)
W = V1 * V2
V = V1 * V2 * Q

# embedding/restriction V <-> W
embWV = V.embeddings[0]@W.restrictions[0] + V.embeddings[1]@W.restrictions[1]
resWV = W.embeddings[0]@V.restrictions[0] + W.embeddings[1]@V.restrictions[1]

# embedding/restriction V <-> Q
embQV = V.embeddings[2]
resQV = V.restrictions[2]

u, uhat, p = V.TrialFunction()
v, vhat, q = V.TestFunction()

nu = 0.001 # viscosity
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size
def tang(vec):
    return vec - (vec*n)*n


alpha = 4  # SIP parameter
dS = dx(element_boundary=True)
a = BilinearForm ( V, symmetric=True)
a += nu * InnerProduct ( Grad(u), Grad(v) ) * dx
a += nu * InnerProduct ( Grad(u)*n, tang(vhat-v) ) * dS
a += nu * InnerProduct ( Grad(v)*n, tang(uhat-u) ) * dS
a += nu * alpha*order*order/h * InnerProduct(tang(vhat-v), tang(uhat-u)) * dS
a += (-div(u)*q - div(v)*p) * dx
a += 1e3 * nu * InnerProduct(div(u), div(v)) * dx
a.Assemble()

def CreateStokesAPC(W, tau=1, mass=0):
    (u, uhat), (v, vhat) = W.TnT()
    elint = True
    stokesA = BilinearForm(W, condense=elint, store_inner=elint)
    # nu=1
    stokesA += nu * InnerProduct ( Grad(u), Grad(v) ) * dx
    # stokesA += InnerProduct ( u, v ) * dx
    # stokesA += nu * InnerProduct ( uhat, vhat ) * dS
    stokesA += nu * InnerProduct ( Grad(u)*n, tang(vhat-v) ) * dS
    stokesA += nu * InnerProduct ( Grad(v)*n, tang(uhat-u) ) * dS
    # alpha = 3
    stokesA += nu * alpha*order*order/h * InnerProduct(tang(vhat-v), tang(uhat-u)) * dS
    stokesA += 1e3 * max(mass, tau*nu) * InnerProduct(div(u), div(v)) * dx
    if mass > 0:
        stokesA += mass * InnerProduct ( u, v ) * dx

    # start off with a couple standard settings
    pc_opts = {
        "ngs_amg_max_levels": 40,
        "ngs_amg_max_coarse_size": 1,
        "ngs_amg_clev": "inv",
        "ngs_amg_log_level": "extra",
        "ngs_amg_log_level_pc": "extra",
        "ngs_amg_do_test": False,
        "ngs_amg_mg_cycle": "V",
        # "ngs_amg_test_levels": True,
        # "ngs_amg_test_smoothers": False
    }

    # what do we have the coarse AMG-levels preserve??
    # pc_opts["presVecs"] = "RTZ"  # 1 DOF per coarse facet, like Raviart-Thomas-zero 
    # pc_opts["presVecs"] = "P0"   # preserve constants
    pc_opts["presVecs"] = "P1"   # preserve divergence-free P1 functions
    # pc_opts["presVecs"] = "FULL_P1"   # preserve all P1 functions

    # smoothers - use hiptmair smoother
    pc_opts["ngs_amg_sm_type"]       = "hiptmair"
    # normal GS smoother in potential space
    pc_opts["ngs_amg_sm_type_pot"]   = "gs"
    # special block-GS smoother optimized for high-order FEM matrices
    pc_opts["ngs_amg_sm_type_range"] = "dyn_block_gs"


    print("stokes-A #DOFS = ", stokesA.space.ndof)

    # c = NgsAMG.stokes_hdiv_gg_3d(stokesA, **pc_opts)

    # c = Preconditioner(stokesA, "local")
    c = Preconditioner(stokesA, "direct")

    stokesA.Assemble()

    if True:
        evs_A = list(la.EigenValues_Preconditioner(mat=stokesA.mat, pre=c.mat, tol=1e-14))
        if stokesA.space.mesh.comm.rank == 0:
            print("\n----")
            print("STOKES TEST")
            print("--")
            print("min ev. preA\A:", evs_A[:5])
            print("max ev. preA\A:", evs_A[-5:])
            print("cond-nr preA\A:", evs_A[-1]/evs_A[0])

    if elint:
        sHex  = stokesA.harmonic_extension.local_mat
        sHexT = stokesA.harmonic_extension_trans.local_mat
        sii   = stokesA.inner_matrix.local_mat
        siii  = stokesA.inner_solve.local_mat
        Id    = IdentityMatrix(stokesA.mat.height)

        sFull = (Id - sHexT) @ (stokesA.mat + sii) @ (Id - sHex)
        pre = ((Id + sHex) @ c.mat @ (Id + sHexT)) + siii
    else:
        sFull = stokesA.mat
        pre = c.mat

        # ngsglobals.msg_level = 3
        # c.Test()

        if False:
            evs_A = list(la.EigenValues_Preconditioner(mat=sFull, pre=pre, tol=1e-14))
            if stokesA.space.mesh.comm.rank == 0:
                print("\n----")
                print("STOKES FULL TEST")
                print("--")
                print("min ev. preA\A:", evs_A[:5])
                print("max ev. preA\A:", evs_A[-5:])
                print("cond-nr preA\A:", evs_A[-1]/evs_A[0])

    # quit()

    return pre, sFull
    # return GMResSolver(mat=stokesA.mat, pre=c.mat, tol=1e-6, printrates=True, maxiter=3), stokesA

def CreatePressureMassPC(Q, tau=1):
    pressureMass = BilinearForm(Q)
    p,q = Q.TnT()
    # pressureMass += (1/(tau * nu)) * InnerProduct(p, q) * dx
    # ddFac = 1e3 * max(mass, tau*nu)
    pressureMass += (1/(tau * nu)) * InnerProduct(p, q) * dx
    pressurePre = Preconditioner(pressureMass, "local")
    pressureMass.Assemble()
    return pressurePre

def CreateFullStokesPC(tau=1, mass=0):
    preA, stokesA = CreateStokesAPC(W, tau, mass)
    preS = CreatePressureMassPC(Q, tau)


    fullPre = embWV @ preA @ resWV + embQV @ preS @ resQV

    return fullPre

# fullPre = CreateFullStokesPC()
# a.Assemble()

gfu = GridFunction(V)

gfu.vec[:] = 0
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

# invstokes = CGSolver(mat=a.mat, pre=fullPre, precision=1e-6)

invstokes = GMResSolver(mat=a.mat, pre=CreateFullStokesPC(), tol=1e-12, printrates=True, maxiter=200)
t = Timer("solve")
t.Start()
# invstokes = a.mat.Inverse(V.FreeDofs())
gfu.vec.data += invstokes @ -a.mat * gfu.vec
t.Stop()

print(f"TIME SOLVE = {t.time}")

quit()

VL2 = VectorL2(mesh, order=order, piola=True)
uL2, vL2 = VL2.TnT()
bfmixed = BilinearForm ( vL2*u*dx, nonassemble=True )
bfmixedT = BilinearForm ( uL2*v*dx, nonassemble=True)

vel = gfu.components[0]
convL2 = BilinearForm( (-InnerProduct(Grad(vL2) * vel, uL2)) * dx, nonassemble=True )
un = InnerProduct(vel,n)
upwindL2 = IfPos(un, un*uL2, un*uL2.Other(bnd=uin))

dskel_inner  = dx(skeleton=True)
dskel_bound  = ds(skeleton=True)

convL2 += InnerProduct (upwindL2, vL2-vL2.Other()) * dskel_inner
convL2 += InnerProduct (upwindL2, vL2) * dskel_bound

class PropagateConvection(BaseMatrix):
    def __init__(self,tau,steps):
        super(PropagateConvection, self).__init__()
        self.tau = tau; self.steps = steps
        self.h = V.ndof; self.w = V.ndof # operator domain and range
        self.mL2 = VL2.Mass(Id(mesh.dim)); self.invmL2 = self.mL2.Inverse()
        self.vecL2 = bfmixed.mat.CreateColVector() # temp vector
    def Mult(self, x, y):
        self.vecL2.data = self.invmL2 @ bfmixed.mat * x # <- project from Hdiv to L2
        for i in range(self.steps):
            self.vecL2.data -= self.tau/self.steps * self.invmL2 @ convL2.mat * self.vecL2
        y.data = bfmixedT.mat * self.vecL2
    def CreateColVector(self):
        return CreateVVector(self.h)
t = 0; tend = 0
tau = 0.01; substeps = 10

m = BilinearForm(u * v * dx, symmetric=True).Assemble()

mstar = m.mat.CreateMatrix()
mstar.AsVector().data = m.mat.AsVector() + tau * a.mat.AsVector()

fullPre = CreateFullStokesPC(tau, 1)
# inv = GMResSolver(mat=mstar, pre=fullPre, tol=1e-6, printrates=True, maxiter=200)
inv = MinResSolver(mat=mstar, pre=fullPre, tol=1e-6, printrates=True, maxiter=200)
# inv = mstar.Inverse(V.FreeDofs(), inverse="umfpack")

Draw(gfu.components[0][0], mesh, "u_x")

input()

tend += 10
res = gfu.vec.CreateVector()
convpropagator = PropagateConvection(tau,substeps)
while t < tend:
    gfu.vec.data += inv @ (convpropagator - mstar) * gfu.vec
    t += tau
    print ("\r  t =", t, end="")
    Redraw()
    quit()

# gfs = []

# for l in range(3):
#     gfs.append(GridFunction(W))
#     c.GetBF(level=3, dof=l, vec=gfs[-1].vec)
#     Draw(gfs[-1].components[0], mesh, f"BF{l}")
#     Draw(div(gfs[-1].components[0]), mesh, f"div(BF{l})")
#     print(f"BF {l} Norm = {InnerProduct(gfs[-1].vec, gfs[-1].vec)**0.5}")

# # L = VectorL2(mesh, order=order)
# # E = ConvertOperator(spacea=V1, spaceb=L)
# # gfl = GridFunction(L)
# # gfl.vec.data = E * gfu.components[0].vec


# # Draw(x, mesh, "x")
# # Draw(div(gfu.components[0]), mesh, "div(BF)")
