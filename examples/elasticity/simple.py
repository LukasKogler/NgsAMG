from ngsolve import *
from ngsolve.meshes import MakeStructured3DMesh

import NgsAMG
from amg_utils import *


# run in 2d or 3d
do3D=True
parallel=True

if do3D:
    geo=unit_cube
    maxh=0.025
    rhsCF = (0, 0, z)
else:
    geo=unit_square
    maxh=0.01
    rhsCF=(0, y)

if parallel:
    # set up simple elasticity problem
    import mpi4py.MPI as MPI
    _, mesh = gen_ref_mesh(geo, maxh=maxh, comm=MPI.COMM_WORLD, nref=0)
else:
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))


YMod   = 1e3    # Young's modulus
PRatio = 0.15   # Poisson ratio
V, a, f = setup_elast(mesh, E=YMod, nu=PRatio, f_vol = CoefficientFunction(rhsCF), diri="left")
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
pc_opts["ngs_amg_sm_steps"] = 2

# overwrite that for level 0 to do 1 x GS
pc_opts["ngs_amg_sm_type_spec"] =  ["gs"]
pc_opts["ngs_amg_sm_steps_spec"] = [ 1 ]

# output/logging
logLevel = ["none", "basic", "normal", "extra"][1]
pc_opts["ngs_amg_log_level"]    = logLevel # output for AMG-level setup
pc_opts["ngs_amg_log_level_pc"] = logLevel # output for smoothers, etc.
pc_opts["ngs_amg_do_test"] = True         # perform preconditioner-test at end of setup

# define preconditioner
pc_class = NgsAMG.elast_3d if do3D else NgsAMG.elast_2d

c = pc_class(a, **pc_opts)

# solve
gfu = Solve(a, f, c, ms=100, tol=1e-12, threading=False)

