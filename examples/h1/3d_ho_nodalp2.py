#
# 3D Poisson, high order, using AMG on P2
#

from ngsolve import *
import ngs_amg

from ngsolve.krylovspace import CGSolver

from amg_utils import *

comm = mpi_world

geo, mesh = gen_cube(maxh=0.1, nref=0, comm=comm)
V, a, f = setup_poisson(mesh, order=3, fes_opts = {"nodalp2" : True}, diri = "bot|right")
gfu = GridFunction(V)

ngsglobals.msg_level = 3

# Now AMG has to do the coarsening on P1 and P2 DOFs, but not on
# the other higher order DOFs, where we only do Gauss-Seidel on the finest level.
#  "on_dofs" : "select"   -- tells AMG that we will provide wich DOFs the coarsening is performed on
#                  [ another option is "range", combined with "lower" and "upper", which will use all DOFs in [lower, upper)
#                    but that does not help us here because of the DOF sorting]
#  "subset" : "nodalp2"   -- tells AMG to use the "special subsets" nodalp2 that is commonly used
#                  [ default is "subset" : "free", which takes all free dofs]
pc_opts = { "ngs_amg_on_dofs" : "select",
            "ngs_amg_subset" : "nodalp2",
            "ngs_amg_max_coarse_size" : 5 }
c = Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)

with TaskManager():
    a.Assemble()
    f.Assemble()
    cb = None if comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
    cg = CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=50, tol=1e-6)
    cg.Solve(sol=gfu.vec, rhs=f.vec)
    print("nits", cg.iterations)
