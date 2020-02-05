import ngsolve, ngs_amg, sys
from amg_utils import *

rots = True
reo = True
nodalp2 = True
use_bddc = True
order = 3
ho_wb = False
ms = 100

print('======= test 3d, ho, rots =', rots, ', reorder=', reo, ', nodalp2=', nodalp2, ', use_bddc=', use_bddc, ', order=', order, 'ho_wb=', ho_wb)
# if reo is not False:
#     if (order==2) and (nodalp2 is False):
#         raise "order 2 + reorder only with nodalp2 (ho smoothing + reorder does now work)"
#     elif order>=3:
#         raise "ho smoothing does not work with reorder!"
if use_bddc and order==2 and nodalp2:
    raise "bddc makes no sense here!"
if reo == "sep" and rots == False:
    raise "reo sep only with rots"
sys.stdout.flush()
geo, mesh = gen_beam(dim=3, maxh=0.3, lens=[5,1,1], nref=0, comm=ngsolve.mpi_world)
fes_opts = dict()
if nodalp2:
    if order>=2:
        fes_opts["nodalp2"] = True
elif use_bddc and not ho_wb: # do not take edge-bubbles to crs space
    fes_opts["wb_withedges"] = False
    # fes_opts["wb_withoutedges"] = True
print('NV: ', mesh.nv)
print('Nedge: ', mesh.nedge)
sys.stdout.flush()
V, a, f = setup_elast(mesh, order=order, fes_opts = fes_opts, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0) ), diri="left", multidim = False, reorder = reo)
ngsolve.ngsglobals.msg_level = 5
pc_opts = { "ngs_amg_max_coarse_size" : 15 }
#pc_opts["ngs_amg_max_levels"] = 2
pc_opts["ngs_amg_do_test"] = True
pc_opts["ngs_amg_edge_thresh"] = 0.02
# rotations or no rotataions?
if rots:
    pc_opts["ngs_amg_rots"] = True
# subset of the low order space    
if use_bddc:
        pc_opts["ngs_amg_on_dofs"] = "select"
        pc_opts["ngs_amg_subset"] = "free"
elif nodalp2:
    if order==2:
        pc_opts["ngs_amg_lo"] = False
    else:
        pc_opts["ngs_amg_on_dofs"] = "select"
        pc_opts["ngs_amg_subset"] = "nodalp2"
# ordering of DOFs within subset


if reo is not False:
    if not rots:
        pc_opts["ngs_amg_dof_blocks"] = [3]
    elif reo == "sep":
        pc_opts["ngs_amg_dof_blocks"] = [3,3]
    else:
        pc_opts["ngs_amg_dof_blocks"] = [6]

print("pc_opts: ", pc_opts)
sys.stdout.flush()

if use_bddc:
    c = ngsolve.Preconditioner(a, "bddc", coarsetype="ngs_amg.elast3d", **pc_opts)
else:
    c = ngsolve.Preconditioner(a, "ngs_amg.elast3d", **pc_opts)


Solve(a, f, c, ms=ms)

print('======= completed test 3d, ho, rots =', rots, ', reorder=', reo, ', nodalp2=', nodalp2, ', use_bddc=', use_bddc, ', order=', order, 'ho_wb=', ho_wb)
sys.stdout.flush()


