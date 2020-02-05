import ngsolve, ngs_amg
import ngs_petsc as petsc
from amg_utils import *

from ngsolve import x, y, z

def Solve(a, f, petsc_opts, kvecs):
    ngs.ngsglobals.msg_level = 5
    gfu = ngs.GridFunction(a.space)
    with ngs.TaskManager():
        a.Assemble()
        f.Assemble()
        ngs.ngsglobals.msg_level = 5
        awrap = petsc.PETScMatrix(a.mat, V.FreeDofs(), format=petsc.PETScMatrix.AIJ)
        awrap.SetNearNullSpace(kvecs)

        c = petsc.KSP(awrap, finalize=True, name="msaKSP", petsc_options = petsc_opts)

        ngs.mpi_world.Barrier()
        ts = ngs.mpi_world.WTime()
        gfu.vec.data = c * f.vec
        ngs.mpi_world.Barrier()
        ts = ngs.mpi_world.WTime() - ts

        print("ndof ", a.space.ndofglobal)
        print("dofs / (sec * np) = ", (a.space.ndofglobal * a.space.dim) / (ts * max(ngs.mpi_world.size-1,1)))

        return gfu

    

def rbms(dim, V, rots):
    if rots:
        raise 'rots'
        tups = [(1, 0, 0), (0, 1, 0), (0,0,1),
                (y, -x, 1)]
    else:
        tups = [(1, 0, 0), (0, 1, 0), (0,0,1),
                (y, -x, 0), (z, 0, -x), (0, z, -y)]
    cfs = [ngsolve.CoefficientFunction(x) for x in tups]
    gf = ngsolve.GridFunction(V)
    kvecs = [gf.vec.CreateVector() for c in cfs]
    for vec, cf in zip(kvecs, cfs):
        gf.Set(cf)
        vec.data = gf.vec
    return kvecs


maxh = 0.15
rots = False
geo, mesh = gen_beam(dim=3, maxh=maxh, lens=[10,1,1], nref=2, comm=ngsolve.mpi_world)
V, a, f = setup_elast(mesh, rotations = False, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0.002) ), diri="left")
pco = { "ksp_type" : "cg",
        "ksp_view" : "ascii:kspv",
        "ksp_monitor" : "",
        "ksp_compute_eigenvalues" : "",
        "ksp_view_eigenvalues" : "",
        "pc_type" : "gamg",
        "pc_gamg_type" : "agg",
        "pc_gamg_agg_nsmooths" : "1",
        "pc_mg_log" : "",
        "ksp_converged_use_initial_residual_norm" : "",
        "ksp_norm_type" : "natural", # preconditioned unpreconditioned none natural
        "ksp_rtol" : "1e-6",
        "pc_mg_cycle_type" : "v"}

pco2 = {"ksp_type" : "cg",
        "ksp_monitor" : "",
        "ksp_compute_eigenvalues" : "",
        "ksp_view_eigenvalues" : "",
        #"mg_levels" : "2",
        "pc_type" : "gamg",
        #"pc_gamg_threshold" : "1e-6",
        "pc_gamg_type" : "agg",
        "pc_gamg_agg_nsmooths" : "1",
        "pc_mg_log" : "",
        #"pc_mg_levels" : "5",
        "pc_mg_cycle_type" : "v",
        "pc_mg_distinct_smoothup" : "",
        # #"mg_coarse_pc_type" : "sor",
        # "mg_coarse_up_pc_type" : "sor",
        # #"mg_coarse_down_pc_type" : "bjac",
        # "mg_coarse_pc_sor_local_forward" : "",
        # "pc_mg_log" : "",
        # "mg_coarse_ksp_type" : "preonly",
        #"mg_coarse_pc_type" : "lu",
        #"mg_coarse_ksp_converged_reason" : "",
        "mg_levels_up_ksp_type" : "richardson",
        "mg_levels_up_pc_type" : "sor",
        "mg_levels_up_pc_sor_local_forward" : "",
        "mg_levels_up_ksp_max_it" : "1",
        # "mg_levels_ksp_type" : "preonly",
        "mg_levels_ksp_type" : "richardson",
        "mg_levels_pc_type" : "sor",
        "mg_levels_pc_sor_local_backward" : "",
        "mg_levels_ksp_max_it" : "1",
        "ksp_view" : "ascii:kspv",
        "ksp_converged_use_initial_residual_norm" : "",
        "ksp_norm_type" : "natural", # preconditioned unpreconditioned none natural
        "ksp_rtol" : "1e-6",
        # "help" : "",
        "ksp_converged_reason" : ""}

gfu = Solve(a, f, petsc_opts = pco2, kvecs = rbms(2, V, rots))
