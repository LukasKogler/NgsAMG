import mpi4py.MPI as MPI

from netgen.csg import unit_cube
from netgen.geom2d import unit_square

from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh

import NgsAMG
from amg_utils import *

from ngsPETSc import KrylovSolver, NullSpace, PETScPreconditioner

# run in 2d or 3d
do3D=True

# beam of length 1, thickness 1/N
N      = 10
numEls = 10

if do3D:
    rhsCF = (0, x, 0)
    ngmesh = MakeStructured3DMesh(hexes=False, nx=numEls*N, ny=numEls, nz=numEls, secondorder=False,
                                  mapping = lambda x,y,z : (x,y/N,z/N)).ngmesh
    diri="back"
    rb_funcs = [(1,0,0), (0,1,0), (0,0,1), (-y,x,0), (0,x,-z), (z,0,-x)]
else:
    rhsCF=(0, x)
    ngmesh = MakeStructured2DMesh(quads=False, nx=numEls*N, ny=numEls, secondorder=False,
                                  mapping = lambda x,y : (x,y/N)).ngmesh
    diri="left"
    rb_funcs = [(1,0),(0,1),(-y,x)]
    

mesh = distribute_mesh(ngmesh, MPI.COMM_WORLD)

YMod   = 1e3    # Young's modulus
PRatio = 0.15   # Poisson ratio
V, a, f = setup_elast(mesh, order=2, E=YMod, nu=PRatio, f_vol = CoefficientFunction(rhsCF), diri=diri, fes_opts={"nodalp2": True})
u, v = V.TnT()


# options for the AMG
pc_opts = { "ngs_amg_max_levels" : 30,
            "ngs_amg_max_coarse_size": 1,
            "ngs_amg_clev": "none",
            "ngs_amg_crs_alg" : ["mis", "spw"][0] }

# prolongation
pc_opts["ngs_amg_prol_type"] =  ["piecewise", "aux_smoothed", "semi_aux_smoothed"][2]
pc_opts["ngs_amg_sp_omega"]  =  0.8
pc_opts["ngs_amg_sp_max_per_row"] = 5

# as smoother, use 2 x block-GS
pc_opts["ngs_amg_sm_type"] =  "bgs"
pc_opts["ngs_amg_sm_steps"] = 1

# output/logging
logLevel = ["none", "basic", "normal", "extra"][1]
pc_opts["ngs_amg_log_level"]    = logLevel # output for AMG-level setup
pc_opts["ngs_amg_log_level_pc"] = logLevel # output for smoothers, etc.
pc_opts["ngs_amg_do_test"] = False         # perform preconditioner-test at end of setup

# integration of nodalP2 into AMG
if True:
    # special nodalP2-embedding  [[ EXPERIMENTAL ]]
    pc_opts["ngs_amg_lo"]               = not do3D # per default, picks out low-order DOFs
    pc_opts["ngs_amg_dof_ordering"]     = "p2Emb"
    pc_opts["ngs_amg_smooth_after_emb"] = False
    pc_opts["ngs_amg_use_emb_sp"]       = False
    pc_opts["ngs_amg_force_ass_flmat"]  = False
else:
    # treats nodalP2-DOFs as vertices for the purpose of coarsening
    pc_opts["ngs_amg_on_dofs"] = "select"
    pc_opts["ngs_amg_subset"]  = "nodalp2"


gfu = GridFunction(V)

rTol=1e-12

# f.Assemble()

ksp_opts = {"ksp_type": "cg",
            "pc_monitor": "",
            "ksp_monitor": "",
            "ksp_rtol": rTol}

usePETSc = True

if usePETSc:
    rbms = []
    for val in rb_funcs:
        rbm = GridFunction(V)
        rbm.Set(CF(val))
        rbms.append(rbm.vec)

    f.Assemble()

    nullspace = NullSpace(V, rbms, near=True)

    tSup = Timer("PETSc - setup")

    tSup.Start()
    a.Assemble() # include a-assemble in both versions!
    c = PETScPreconditioner(a.mat, V.FreeDofs(), nullspace=nullspace,
        solverParameters={"pc_type": "gamg"})
    tSup.Stop()

else:
    # define preconditioner
    pc_class = NgsAMG.elast_3d if do3D else NgsAMG.elast_2d

    c = pc_class(a, **pc_opts)
    # c = Preconditioner(a, "local")

    # ksp_opts["pc_type"] = "mat"
    # ksp_opts["ngs_mat_type"] = "python"


    # # solve

gfu = Solve(a, f, c, ms=100, tol=rTol, threading=False, needsAssemble=not usePETSc, printTimers=not usePETSc)

if usePETSc:
    print(f"PETSc - setup = {tSup.time}")


