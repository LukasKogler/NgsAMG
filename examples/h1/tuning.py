#
# 3D Poisson, some tuning parameters
#

from ngsolve import *
import ngs_amg

from ngsolve.krylovspace import CGSolver

from amg_utils import *

comm = NG_MPI_world

geo, mesh = gen_cube(maxh=0.05, nref=0, comm=comm)

V, a, f = setup_poisson(mesh)
gfu = GridFunction(V)


## This is a non-comprehensive list of options for NgsAMG. (For now it is also the only documentation for that, sorry)
## most of them work for H1 or Elasticity. However, keep in ming that one "vertex" stands for one DOF in the scalar H1 case,
## and for 2/3 displacements and 1/3 rotations for elasticity. (so either 1, 3 or 6 DOFs)
pc_opts = dict()

# All options for NgsAMG have to be prefixed with "ngs_amg_".
pf = "ngs_amg_"

# Log-level
pc_opts[pf + "log_level"] = "normal"  # how much info to collect [none, basic, normal, extra]
pc_opts[pf + "log_file"]  = "amg.log" # file to write the summary to
pc_opts[pf + "print_log"] = True      # wether to print summary to shell on rank 0


# Coarsening rate
pc_opts[pf + "max_levels"] = 10       # assemble max. this many levels
pc_opts[pf + "first_aaf"] = 0.05      # assemble first level after reducing "verices" by this factor
pc_opts[pf + "aaf"] = 0.1             # after the first time, assemble after this factor
pc_opts[pf + "aaf_scale"] = 10        # scale "aaf" with this factor. (We assemble after first_aaf, aaf, aaf_scale*aaf, aaf_scale**2*aaf, ...)


# Coarsening algorithm
pc_opts[pf + "energy"] = "alg"        # how to define coarsening weights. "alg" .. from assembled mat,
                                      # "elmat" .. from element matrices, "triv" .. dummy
pc_opts[pf + "edge_thresh"] = 0.05    # threshold for coarsening along edge, some value between 0 and 1
pc_opts[pf + "vert_thresh"] = 0.2     # threshold for dropping masss dominated vertices, some value between 0 and 1


# Enble smoothed prolongation. On by default.
# Increases cost, improves condition number (drastically)
pc_opts[pf + "enable_sp"] = True
pc_opts[pf + "sp_max_per_row"] = 4     # maximum non-zeros per row for smoothed prolongation (for elasticity, matrix-valued entries!)
pc_opts[pf + "sp_thresh"] = 0.1        # strength-threshhold for entries in smoothed prolongation, weak entries are dropped
pc_opts[pf + "sp_omega"] = 1           # the omega in P = (I - omega D^{-1} A) Phat


# Enable redistributing on coarse levels. On by default.
# Very important for effciency (and condition) for large computations
# first_rdaf, rdaf, rdaf_scale as for aaf above
pc_opts[pf + "enable_redist"] = True
pc_opts[pf + "rd_min_nv"] = 4000       # try to re-ditribute such that we always have at least this many vertices per proc
pc_opts[pf + "rd_min_nv_thr"] = 2000   # re-ditribute when fewer than this many vertices per proc are left (should be <= 0.5 * rd_min_nv_thr)
pc_opts[pf + "rd_seq"] = 4000          # after reaching this amount of vertices, re-distribute to a single proc


# Only supported for H1. Rebuilds mesh topology on coarse levels. On by default for H1.
# Improves condition for problems with a couple million unknowns or more.
# Increases cost a bit. Experimental.
pc_opts[pf + "enable_rbm"] = True


# Smoothers. These are all implementations of Hybrid Gauss-Seidel.
# That is, basically Gauss-Seidel locally everywhere and some cheating
# on subdomain-boundaries to get something convergent.
pc_opts[pf + "sm_ver"] = 3             # which version, 1, 2 or 3
                                       # versions 2 and 3 are faster than 1, but need a bit more memory
                                       # version 3 can overlap MPI communication and computation
                                       # withoug overlap, 2 is probably a bit faster than 3
pc_opts[pf + "sm_NG_MPI_overlap"] = True  # try to overlap MPI and local communication (version 3 only)
pc_opts[pf + "sm_NG_MPI_thread"]  = False # do MPI communication in seperate thread (version 3 only)

# In general, I suggest:
#   - [overlap=True, thread=False] with Infiniband/Omnipath/etc. or an MPI implementation that actually
#     progresses communication in the background (OpenMPI does, in my experience, not)
#   - [overlap=True, thread=True] with hyper-threading or multiple cores per proc
#     the MPI library needs to support NG_MPI_THREAD_MULTIPLE
#   - [overlap=False, thread=..] otherwise, but in that case maybe "sm_ver"=2 might be faster anyways

c = Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)

with TaskManager():
    a.Assemble()
    f.Assemble()
    cb = None if comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
    cg = CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=50, tol=1e-6)
    cg.Solve(sol=gfu.vec, rhs=f.vec)
    print("nits", cg.iterations)
