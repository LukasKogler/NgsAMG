#
# 3D Poisson, low order - show some tuning parameters
#

from ngsolve import *
import ngs_amg

from ngsolve.krylovspace import CGSolver

from amg_utils import *

comm = mpi_world

geo, mesh = gen_cube(maxh=0.2, nref=0, comm=comm)


V, a, f = setup_poisson(mesh, order=3, fes_opts = { "wb_withoutedges" : False }, diri="left")
gfu = GridFunction(V)

u,v = V.TnT()
f2 = LinearForm(V)
f2 += SymbolicLFI(1*v)
    
ngsglobals.msg_level = 4

# This will not work particularly well !
# In 3D, bddc needs edge-bubbles int he coarse space, but AMG cannot use those for coarsening,
# therefore only does smoothing on those bubbles. See 3d_bddc_nodalp2 for an improvement
pc_opts = { "ngs_amg_max_coarse_size" : 10,
            "ngs_amg_log_level" : "extra",
            "ngs_amg_clev" : "none",
            "ngs_amg_do_test" : True }
c = Preconditioner(a, "bddc", coarsetype = "ngs_amg.h1_scal", **pc_opts)

with TaskManager():
    a.Assemble()
    f.Assemble()
    cb = None #if comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
    cg = CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=500, tol=1e-6)
    cg.Solve(sol=gfu.vec, rhs=f.vec)
    print("nits", cg.iterations)

c.Test()
