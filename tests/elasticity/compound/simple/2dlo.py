import ngsolve, ngs_amg, sys
from amg_utils import *

rots = False
reo = True#"u/r"
ms = 50
maxh = 0.2
mdim = True
order = 2

print('======= test 2d, lo, rots =', rots, ', reorder=', reo)
sys.stdout.flush()
geo, mesh = gen_beam(dim=2, maxh=0.2, lens=[6,1], nref=0, comm=ngsolve.mpi_world)

V, a, f = setup_elast(mesh, order=order, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005) ), diri="left", multidim = mdim, reorder = reo)
print('V ndof', V.ndof)
ngsolve.ngsglobals.msg_level = 5
pc_opts = { "ngs_amg_max_coarse_size" : 10 }
if not mdim:
    if reo is "u/r":
        pc_opts["ngs_amg_dof_blocks"] = [2,1] if rots else [2]
    else:
        pc_opts["ngs_amg_dof_blocks"] = [3] if rots else [2]
if rots:
    pc_opts["ngs_amg_rots"] = True


# 2d:
pc_opts["ngs_amg_rots"] = False
pc_opts["ngs_amg_dof_blocks"] = [1,1]
c = ngs_amg.elast_2d(a, **pc_opts)

Solve(a, f, c, ms=ms)
quit()
# a.Assemble()

print('AMAT', a.mat)

print('======= completed test 2d, lo, rots =', rots, ', reorder=', reo, '\n\n')
sys.stdout.flush()

from bftester_vec import shape_test
shape_test(mesh, maxh, V, a, c, 3, 3 if mdim else 1, reo=reo)
