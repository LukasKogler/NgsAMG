#
# 3D Poisson, high order, using AMG + BDDC
#

from ngsolve import *
import ngs_amg

from ngsolve.krylovspace import CGSolver

from amg_utils import *

comm = NG_MPI_world

geo, mesh = gen_cube(maxh=0.2, nref=0, comm=comm)

# We again use nodalp2 (see 2d_nodalp2.py)
V, a, f = setup_poisson(mesh, order=5, fes_opts = { "nodalp2" : True })

print("ndofs ", V.ndof, V.lospace.ndof)
gfu = GridFunction(V)

# We again have to tell AMG which DOFs to perform the coarsening on
# BDDC already filters out non-wirebasket and dirichlet-DOFs and the coarse level
# matrix handed to AMG looks like this:
#        A_FF 0
#         0   0
# (F stands wor wirebasket+free)
# We tell AMG to use the bitarray it is given by BDDC to define the subset
# for the coarsening:
#  "subset" : "free"
ngsglobals.msg_level = 4
pc_opts = { "ngs_amg_on_dofs" : "select",
            "ngs_amg_do_test" : True,
            "ngs_amg_subset" : "free",
            "ngs_amg_max_coarse_size" : 5 }
c = Preconditioner(a, "bddc", coarsetype = "ngs_amg.h1_scal", **pc_opts)
# c = Preconditioner(a, "bddc")#, coarsetype = "ngs_amg.h1_scal", **pc_opts)

with TaskManager():
    a.Assemble()
    f.Assemble()
    cb = None #if comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
    cg = CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=500, tol=1e-6)
    cg.Solve(sol=gfu.vec, rhs=f.vec)
    print("nits", cg.iterations)
