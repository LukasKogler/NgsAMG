from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh

import NgsAMG
from amg_utils import *


rhsCF = (0, x, 0)
mesh = MakeStructured3DMesh(hexes=False, nx=20, ny=3, nz=3, secondorder=False,
                            mapping = lambda x,y,z : (10*x, y, z))
diri="back"
    


YMod   = 1e3    # Young's modulus
PRatio = 0.15   # Poisson ratio
# diri=""
V, a, f = setup_elast(mesh, E=YMod, nu=PRatio, f_vol = CoefficientFunction(rhsCF), diri=diri, order=1)
u, v = V.TnT()
# a += 1e-6 * InnerProduct(u, v) * dx

# options for the AMG
pc_opts = { "ngs_amg_max_levels" : 20,
            "ngs_amg_max_coarse_size": 1,
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

# logging
logLevel = ["none", "basic", "normal", "extra"][2]
pc_opts["ngs_amg_log_level"]    = logLevel # output for AMG-level setup
pc_opts["ngs_amg_log_level_pc"] = logLevel # output for smoothers, etc.



c = NgsAMG.elast_3d(a, **pc_opts)

a.Assemble()

gfu = GridFunction(V)

# visualization for nodalp2 does not work correctly, draw with normal H1
VV = H1(mesh, order=1, dim=mesh.ngmesh.dim)


L=2
D=3
C=3   #0,1,2 are x/y/z disp, 3,4,5 are x/y/z rotations 

# get BF from level L, component C
c.GetBF(vec=gfu.vec, level=L, dof=D, comp=C)
gfVis = GridFunction(VV)
gfVis.Set(gfu)
Draw(gfVis, VV.mesh, "BF")
