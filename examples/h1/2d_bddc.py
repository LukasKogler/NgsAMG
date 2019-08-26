#
# 2D Poisson, high order, using AMG + BDDC
#

from ngsolve import *
import ngs_amg

from ngsolve.krylovspace import CGSolver

from amg_utils import *

comm = mpi_world

geo, mesh = gen_square(maxh=0.05, nref=0, comm=mpi_world)
V, a, f = setup_poisson(mesh, order=4, diri="right")
gfu = GridFunction(V)

# Here we use the BDDC precodnitioner, and tell it to plug the coarse level matrix into AMG.
pc_opts = { "ngs_amg_max_coarse_size" : 5 }
c = Preconditioner(a, "bddc", coarsetype = "ngs_amg.h1_scal", **pc_opts)

with TaskManager():
    a.Assemble()
    f.Assemble()
    cb = None if comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
    cg = CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=100, tol=1e-12)
    cg.Solve(sol=gfu.vec, rhs=f.vec)
    print("nits", cg.iterations)
