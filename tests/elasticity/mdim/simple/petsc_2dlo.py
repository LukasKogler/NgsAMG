import ngsolve, ngs_amg
import ngs_petsc as petsc
from amg_utils import *

from ngsolve import x, y

def Solve(a, f, petsc_opts, kvecs):
    ngs.ngsglobals.msg_level = 5
    gfu = ngs.GridFunction(a.space)
    with ngs.TaskManager():
        a.Assemble()
        f.Assemble()
        ngs.ngsglobals.msg_level = 5
        awrap = petsc.PETScMatrix(a.mat, V.FreeDofs(), format=petsc.PETScMatrix.AIJ)
        awrap.SetNearNullSpace(kvecs)
        c = petsc.KSP(awrap, finalize=True, name="msaKSP", petsc_options = petsc_opts)
        gfu.vec.data = c * f.vec
        return gfu


def rbms(dim, V, rots):
    if rots:
        tups = [(1, 0, 0), (0, 1, 0), (y, -x, 1)]
    else:
        tups = [(1, 0), (0, 1), (y, -x)]
    cfs = [ngsolve.CoefficientFunction(x) for x in tups]
    gf = ngsolve.GridFunction(V)
    kvecs = [gf.vec.CreateVector() for c in cfs]
    for vec, cf in zip(kvecs, cfs):
        gf.Set(cf)
        vec.data = gf.vec
    return kvecs

maxh = 0.05
rots = False
geo, mesh = gen_beam(dim=2, maxh=maxh, lens=[5,1], nref=1, comm=ngsolve.mpi_world)
V, a, f = setup_elast(mesh, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005) ), diri="left")
pco = { "ksp_type" : "cg",
        "ksp_view" : "ascii:kspv",
        "ksp_monitor" : "",
        "ksp_compute_eigenvalues" : "",
        "ksp_view_eigenvalues" : "",
        "pc_type" : "gamg",
        "pc_gamg_type" : "agg",
        "pc_gamg_agg_nsmooths" : "1",
        "pc_mg_log" : "",
        "pc_mg_cycle_type" : "v"}
gfu = Solve(a, f, petsc_opts = pco, kvecs = rbms(2, V, rots))
