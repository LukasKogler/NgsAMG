import mpi4py.MPI as MPI

from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh

import NgsAMG
from amg_utils import *


# !! EXPERIMENTAL !!

# beam of numEls_L x numEls_T x numEls_T elements,
#   length    is  numElsL
#   thickness is  1/N
#   elements are stretched by factor N/numEls_T
numEls_L = 100
numEls_T = 5
N        = 10

# use nodal-p2 with improved AMG-integration
nodalP2=True

stretchFactor = numEls_T/N

maxh=0.025
rhsCF = (0, x, 0)
ngmesh = MakeStructured3DMesh(hexes=False, nx=numEls_L, ny=numEls_T, nz=numEls_T, secondorder=False,
                              mapping = lambda x,y,z : (numEls_L * x, stretchFactor*y,stretchFactor*z)).ngmesh
diri="back"
    

mesh = distribute_mesh(ngmesh, MPI.COMM_WORLD)


YMod   = 1e3    # Young's modulus
PRatio = 0.15   # Poisson ratio
# diri=""
V, a, f = setup_elast(mesh, E=YMod, nu=PRatio, f_vol = CoefficientFunction(rhsCF), diri=diri, order=2, fes_opts={"nodalp2": nodalP2})
u, v = V.TnT()
# a += 1e-6 * InnerProduct(u, v) * dx

# options for the AMG
pc_opts = { "ngs_amg_max_levels" : 20,
            "ngs_amg_max_coarse_size": 10,
            "ngs_amg_clev": "inv"}

# coarsening - use successive pairwise algorithm, limit number of steps for this 1d-ish problem
pc_opts["ngs_amg_crs_alg"]         =  "spw",
pc_opts["ngs_amg_spw_rounds"]      =  2,
pc_opts["ngs_amg_spw_rounds_spec"] =  [ 3 ]

# prolongation
pc_opts["ngs_amg_prol_type"] =  ["piecewise", "aux_smoothed", "semi_aux_smoothed"][2]
pc_opts["ngs_amg_sp_omega"]  =  0.7
pc_opts["ngs_amg_sp_max_per_row"] = 6

# improve smoothed prol [[ experimental ]]
pc_opts["ngs_amg_sp_improve_its"] = 2


# use block-smoother
pc_opts["ngs_amg_sm_type"] =  "bgs"
pc_opts["ngs_amg_sm_steps"] = 1


# output/logging
logLevel = ["none", "basic", "normal", "extra"][2]
pc_opts["ngs_amg_log_level"]    = logLevel # output for AMG-level setup
pc_opts["ngs_amg_log_level_pc"] = logLevel # output for smoothers, etc.
pc_opts["ngs_amg_do_test"] = True         # perform preconditioner-test at end of setup
# pc_opts["ngs_amg_test_levels"] = True         # perform preconditioner-test at end of setup


# integration of nodalP2 into AMG
if nodalP2:
    if True:
        # special nodalP2-embedding  [[ EXPERIMENTAL ]]
        pc_opts["ngs_amg_lo"]           = False # per default, picks out low-order DOFs
        pc_opts["ngs_amg_dof_ordering"] = "p2Emb"
        pc_opts["ngs_amg_smooth_after_emb"] = False
        # pc_opts["ngs_amg_force_ass_flmat"] = False
    else:
        # treats nodalP2-DOFs as vertices for the purpose of coarsening
        pc_opts["ngs_amg_on_dofs"] = "select"
        pc_opts["ngs_amg_subset"]  = "nodalp2"


c = NgsAMG.elast_3d(a, **pc_opts)

gfu = Solve(a, f, c, ms=150, tol=1e-12, threading=False)

# visualization for nodalp2 does not work correctly, draw with normal H1
VV = H1(mesh, order=order, dim=mesh.ngmesh.dim)
gfVis = GridFunction(VV)
gfVis.Set(gfu)

Draw(gfVis)
Draw(gfVis[1], mesh, "y-disp")

