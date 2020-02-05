import ngsolve, ngs_amg
from amg_utils import *

rots = False
ms = 100
pc_opts = { "ngs_amg_max_coarse_size" : 10 }
order = 3
dobft = False
dosol = True

print('test 3d with nodalp2, order  = ', order, 'rots  = ', rots)
maxh = 0.35
geo, mesh = gen_beam(dim = 3, maxh = maxh, lens = [5,1,1], nref = 0, comm = ngsolve.mpi_world)
fes_opts = dict()
if order > 1:
    fes_opts["nodalp2"] = True
V, a, f = setup_elast(mesh, order = order, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0) ), diri = "left", fes_opts=fes_opts)

print('V ndof', V.lospace.ndof, V.ndof)
print('mesh nn ', mesh.nv, mesh.nedge) 
print('V free ', sum(V.FreeDofs()), "of", V.ndof)
print('bnds ', [ x for x in mesh.GetBoundaries()] )

# W = ngsolve.H1(mesh, order=order, dirichlet="left")
# print('W free ', sum(W.FreeDofs()), "of", W.ndof)


import sys
sys.stdout.flush()

pc_opts["ngs_amg_rots"] = rots
pc_opts["ngs_amg_max_coarse_size"] = 1
if order  ==  2:
    pc_opts["ngs_amg_lo"] = False
elif order > 2:
    pc_opts["ngs_amg_on_dofs"] = "select"
    pc_opts["ngs_amg_subset"] = "nodalp2"
if rots is False:
    pc_opts["ngs_amg_reg_mats"] = True
    pc_opts["ngs_amg_reg_rmats"] = True
c = ngs_amg.elast_3d(a, **pc_opts)

if dosol:
    Solve(a, f, c, ms = ms)
else:
    ngsolve.ngsglobals.msg_level = 5
    a.Assemble()
    ngsolve.ngsglobals.msg_level = 1
    c.Test()


import sys
sys.stdout.flush()
if dobft:
    from bftester_vec import *
    shape_test(mesh, maxh, V, a, c, 3, order=order)
