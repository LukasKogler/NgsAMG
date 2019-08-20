#
# 3D Poisson, low order - show some tuning parameters
#

from ngsolve import *
import ngs_amg

from ngsolve.krylovspace import CGSolver

from amg_utils import *

comm = mpi_world

geo, mesh = gen_cube(maxh=0.1, nref=0, comm=comm)

V, a, f = setup_poisson(mesh, order=4)
gfu = GridFunction(V)

pc_opts = { "ngs_amg_max_coarse_size" : 5 }
c = Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)

with TaskManager():
    a.Assemble()
    f.Assemble()
    cb = None if comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
    cg = CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=50, tol=1e-6)
    cg.Solve(sol=gfu.vec, rhs=f.vec)
    print("nits", cg.iterations)
