#
# 2D Poisson, lowest order
#

from ngsolve import *
import ngs_amg

from ngsolve.krylovspace import CGSolver

from amg_utils import *

comm = NG_MPI_world

geo, mesh = gen_square(maxh=0.05, nref=0, comm=comm)

V, a, f = setup_poisson(mesh, order=1, beta=0, diri=".*")
gfu = GridFunction(V)

ngsglobals.msg_level = 4

# We tell AMG to do coarsening until at most 5 DOFs are left,
# at which point we directlt invert the coarse matrix
pc_opts = { "ngs_amg_max_coarse_size" : 5,
            "ngs_amg_oldsm" : False,
            "ngs_amg_sm_NG_MPI_overlap" : True,
            "ngs_amg_enable_sp" : True,
            "ngs_amg_enable_redist" : True,
            "ngs_amg_do_test" : True,
            "ngs_amg_log_level" : "extra",
            "ngs_amg_print_log" : True }
c = Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)


# We can get information about what AMG is doing by setting the msg_level:
#   <3  .. nothing
#    3  .. a little in the beginning
#    4+ .. some on every level

with TaskManager():
    a.Assemble()
    f.Assemble()
    cb = None if comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
    cg = CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=50, tol=1e-6)
    cg.Solve(sol=gfu.vec, rhs=f.vec)
    print("nits", cg.iterations)
    c.Test()
    
