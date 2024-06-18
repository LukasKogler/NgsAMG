import mpi4py.MPI as MPI

from netgen.csg import unit_cube
from netgen.geom2d import unit_square

from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh

import NgsAMG
from amg_utils import *

from ngsPETSc import KrylovSolver, NullSpace

# run in 2d or 3d
do3D=False

# beam of length 1, thickness 1/N
N      = 10
numEls = 30

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
V, a, f = setup_elast(mesh, E=YMod, nu=PRatio, f_vol = CoefficientFunction(rhsCF), diri=diri)
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
pc_opts["ngs_amg_sm_type"] =  "gs"
pc_opts["ngs_amg_sm_steps"] = 1

# overwrite that for level 0 to do 1 x GS
pc_opts["ngs_amg_sm_type_spec"] =  ["gs"]
pc_opts["ngs_amg_sm_steps_spec"] = [ 1 ]

# output/logging
logLevel = ["none", "basic", "normal", "extra"][1]
pc_opts["ngs_amg_log_level"]    = logLevel # output for AMG-level setup
pc_opts["ngs_amg_log_level_pc"] = logLevel # output for smoothers, etc.
pc_opts["ngs_amg_do_test"] = True         # perform preconditioner-test at end of setup


gfu = GridFunction(V)

rTol=1e-12

f.Assemble()

ksp_opts = {"ksp_type": "cg",
            "pc_monitor": "",
            "ksp_monitor": "",
            "ksp_rtol": rTol}

if True:
    rbms = []
    for val in rb_funcs:
        rbm = GridFunction(V)
        rbm.Set(CF(val))
        rbms.append(rbm.vec)

    a.Assemble()

    nullspace = NullSpace(V, rbms, near=True)

    ksp_opts["pc_type"] = "gamg"

    tpSup = Timer("PETSc - setup")
    tpSup.Start()
    ksp = KrylovSolver(a, V, solverParameters=ksp_opts, nullspace=nullspace)
    tpSup.Stop()

    tpSol = Timer("PETSc - solve")
    tpSol.Start()
    ksp.solve(f.vec, gfu.vec)
    tpSol.Stop()

    print(f"PETSC - setup {tpSup.time}")
    print(f"PETSC - solve {tpSol.time}")
else:
    # define preconditioner
    pc_class = NgsAMG.elast_3d if do3D else NgsAMG.elast_2d

    # c = pc_class(a, **pc_opts)
    c = Preconditioner(a, "local")

    ksp_opts["pc_type"] = "mat"
    ksp_opts["ngs_mat_type"] = "python"


    # # solve
    # gfu = Solve(a, f, c, ms=100, tol=rTol, threading=False)

    tpSup = Timer("NgsAMG - setup")
    tpSup.Start()
    a.Assemble()
    tpSup.Stop()

    tpConv = Timer("NgsAMG - setup")
    tpConv.Start()
    ksp = KrylovSolver(a.mat, V.FreeDofs(), p=c.mat, solverParameters=ksp_opts)
    tpConv.Stop()

    # quit()
    tpSol = Timer("NgsAMG - solve")
    tpSol.Start()
    ksp.solve(f.vec, gfu.vec)
    tpSol.Stop()

    print(f"NgsAMG - setup   {tpSup.time}")
    print(f"NgsAMG - convert {tpSup.time}")
    print(f"NgsAMG - solve   {tpSol.time}")


    Draw(gfu)

