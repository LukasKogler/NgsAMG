import ngsolve, ngs_amg, sys
from amg_utils import *

ro = False
rots = False

print('======= test 3d, lo, rots =', rots, ', reorder=', reo)
sys.stdout.flush()
geo, mesh = gen_beam(dim=3, maxh=0.2, lens=[7,1,1], nref=0, comm=ngsolve.mpi_world)
V, a, f = setup_elast(mesh, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0) ), diri="left", multidim = False, reorder = reo)
print('V ndof', V.ndof)
ngsolve.ngsglobals.msg_level = 5
pc_opts = { "ngs_amg_max_coarse_size" : 10 }
if reo == "sep":
    pc_opts["ngs_amg_dof_blocks"] = [3,3] if rots else [3]
elif reo is not False:
    pc_opts["ngs_amg_dof_blocks"] = [6] if rots else [3]
if rots:
    pc_opts["ngs_amg_rots"] = True
c = ngsolve.Preconditioner(a, "ngs_amg.elast3d", **pc_opts)
Solve(a, f, c, ms=ms)
print('======= completed test 3d, lo, rots =', rots, ', reorder=', reo, '\n\n')
sys.stdout.flush()
