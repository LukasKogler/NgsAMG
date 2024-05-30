import os, sys
from ctypes import CDLL, RTLD_GLOBAL
try:
    CDLL(os.path.join(os.environ["MKLROOT"], "lib/intel64/libmkl_rt.so"), RTLD_GLOBAL)
except:
    try:
        CDLL(os.path.join(os.environ["MKL_ROOT"], "lib/intel64/libmkl_rt.so"), RTLD_GLOBAL)
    except:
        pass

import ngsolve, ngs_amg
from amg_utils import *

def Solve(a, f, c, ms = 100):
    ngs.ngsglobals.msg_level = 5
    gfu = ngs.GridFunction(a.space)
    with ngs.TaskManager():
        a.Assemble()
        f.Assemble()
        ngs.ngsglobals.msg_level = 1
        if c is None:
            c = a.mat.Inverse(V.FreeDofs())
        else:
            c.Test()
        quit()
        cb = None if a.space.mesh.comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
        cg = ngs.krylovspace.CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=ms, tol=1e-6)
        cg.Solve(sol=gfu.vec, rhs=f.vec)
        print("used nits = ", cg.iterations)
        ngsolve.Draw(mesh, deformation=ngsolve.CoefficientFunction((gfu[0], gfu[1], 0)))

        
maxh = 0.05
rots = True
geo, mesh = gen_beam(dim=2, maxh=maxh, lens=[5,1], nref=1, comm=ngsolve.mpi_world)
V, a, f = setup_elast(mesh, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005) ), diri="")
u, v = V.TnT()
a += 1e-10 * u * v * ngsolve.dx

print('V ndof', V.ndof)
pc_opts = { "ngs_amg_rots" : rots,
            "ngs_amg_max_levels" : 20,
            "ngs_amg_max_clev" : "inv",
            #"ngs_amg_first_aaf" : 0.1,
            "ngs_amg_log_level" : "extra",
            "ngs_amg_max_coarse_size" : 10,
            "ngs_amg_enable_sp" : True,
            "ngs_amg_sp_max_per_row" : 10,
            "ngs_amg_sm_ver" : 3 }
# c = ngsolve.Preconditioner(a, "ngs_amg.elast2d", **pc_opts)
c = ngs_amg.elast_2d(a, **pc_opts)
Solve(a, f, c, ms=400)

# NOTE: some_flip_with_rots

#a.Assemble()

#from bftester_vec import shape_test
#shape_test(mesh, maxh, V, a, c, 3)
