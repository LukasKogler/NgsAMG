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

maxh = 0.1
geo, mesh = gen_beam(dim=3, maxh=maxh, lens=[1,1,1], nref=1, comm=ngsolve.mpi_world)
V, a, f = setup_elast(mesh, rotations = False, f_vol = ngsolve.CoefficientFunction( (0, -1e-5, -1e-5) ), diri="")
u, v = V.TnT()
a += 1e-7 * u * v * ngsolve.dx

#pc_opts = { "ngs_amg_reg_rmats" : True }
pc_opts = { "ngs_amg_reg_rmats" : True,
            "ngs_amg_max_levels" : 20,
            "ngs_amg_enable_redist" : False,
            "ngs_amg_max_coarse_size" : 20,
            "ngs_amg_log_level" : "extra",
            "ngs_amg_enable_sp" : True,
            "ngs_amg_sp_max_per_row" : 4,
            "ngs_amg_first_aaf" : 0.025}

gfu = ngsolve.GridFunction(V)
a.Assemble()
f.Assemble()
gfu.vec.data = a.mat.Inverse(V.FreeDofs()) * f.vec

#ngsolve.Draw(mesh, deformation = ngsolve.CoefficientFunction((gfu[0], gfu[1], gfu[2])), name="defo")
ngsolve.Draw(mesh, deformation = ngsolve.CoefficientFunction((gfu[0], gfu[1])), name="defo")
ngsolve.Draw(gfu[2], mesh, name="rot")
# for i in range(6):
#     ngsolve.Draw(gfu[i], mesh, name="comp_"+str(i))

# ngsolve.Draw(gfu, name="sol")


# # c = ngsolve.Preconditioner(a, "ngs_amg.elast3d", **pc_opts)
c = ngs_amg.elast_3d(a, **pc_opts)
pt = 0 #100 * 1024 * 1024
with ngsolve.TaskManager(pajetrace = pt):
     Solve(a, f, c, ms=40)

# if ngsolve.mpi_world.rank == 1:
#SetNumThreads(5)


#from bftester_vec import shape_test
#shape_test(mesh, maxh, V, a, c, 6)
