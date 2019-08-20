#
# 3D Poisson, oder 2 with nodal base functions
#

from ngsolve import *
import ngs_amg

from ngsolve.krylovspace import CGSolver

from amg_utils import *

comm = mpi_world

geo, mesh = gen_cube(maxh=0.1, nref=0, comm=comm)

# Per default, the H1 space has piecewise linear hat functions as lowest order
# base functions and edge bubbles as P2 functions. This means that the coefficient
# vector where every entry is a 1 does not give us the constant function!
#
# nodalp2 : True tells the H1-space to replace these with piecewise P2 functions
# that are nodal w.r.t vertices and edge mid-points, so the constant 1 coefficient vector
# gives us the constant function again
#   --> !! nodalp2 has no effect in 2D !! <---
V, a, f = setup_poisson(mesh, order=2, fes_opts = { "nodalp2" : True }, diri="left", alpha=1, beta=0)
gfu = GridFunction(V)

ngsglobals.msg_level = 3

# Per default, we take the low-order part of the matrix and do coarsening on that.
# However, in this case, we want to use all DOFs for the coarsening, so we set
#    amg_lo : False
# This tells the AMG to use all DOFs for coarsening
pc_opts = { "ngs_amg_lo" : False, "ngs_amg_max_coarse_size" : 5 }
c = Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)

with TaskManager():
    a.Assemble()
    f.Assemble()
    cb = None if comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
    cg = CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=50, tol=1e-6)
    cg.Solve(sol=gfu.vec, rhs=f.vec)
    print("nits", cg.iterations)


