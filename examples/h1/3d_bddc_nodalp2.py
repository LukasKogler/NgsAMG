#
# 3D Poisson, high order, using AMG + BDDC
#

from ngsolve import *
import ngs_amg

from ngsolve.krylovspace import CGSolver

from amg_utils import *

comm = mpi_world

geo, mesh = gen_cube(maxh=0.05, nref=0, comm=comm)

# We again use nodalp2 (see 2d_nodalp2.py)
V, a, f = setup_poisson(mesh, order=4)
gfu = GridFunction(V)

# We again have to tell AMG which DOFs to perform the coarsening on
# This time, BDDC already filters out dirichlet-DOFs and the coarse level
# matrix handed to AMG looks like this:
#        A_FF 0
#         0   0
# Here, "F" stands for "wirebasket and free". There are no non-zero entries for
# "wirebasket and dirichlet" DOFs, so we cannot take the "nodalp2" special subset as in
# 2d_ho_nodalp2.py for the coarsening.
#
# Instead, we tell AMG to use the bitarray it is given by BDDC for the coarsening:
#  "subset" : "free"
pc_opts = { "ngs_amg_on_dofs" : "select",
            "ngs_amg_subset" : "free",
            "ngs_amg_max_coarse_size" : 5 }
c = Preconditioner(a, "bddc", coarse_type = "ngs_amg.h1_scal", **pc_opts)

with TaskManager():
    a.Assemble()
    f.Assemble()
    cb = None if comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
    cg = CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=50, tol=1e-6)
    cg.Solve(sol=gfu.vec, rhs=f.vec)
    print("nits", cg.iterations)
