from ngsolve import *
from amg_utils import *
import netgen, sys, ngs_amg
from mcs_utils import *
from ngsolve.la import EigenValues_Preconditioner

# SetTestoutFile('test.out')

RT, order = True, 1
# RT, order = False, 2
geo, maxh = netgen.csg.unit_cube, 0.4
el_int = True
sym = True
# diri,l2c = "back", 0
diri,l2c = "", 1
fcf = CoefficientFunction((0,-x*(1-x),0))

geo, mesh = gen_ref_mesh (geo, maxh, nref = 0, comm=mpi_world, mesh_file = '', save = False)

# mesh = OneTet()

X, a, f = setup_mcs(mesh, order = order, force = fcf, diriN = diri, diriT = diri, el_int = el_int, sym = sym, RT = RT, l2_coef = l2c)

gfu = GridFunction(X)

pc_aux = ngs_amg.mcs3d(a)

a.Assemble()
f.Assemble()


# TestRBMS(X, a.mat, pc_aux, draw=True)


sgf = GridFunction(X)
sgf.vec.data = a.mat.Inverse(X.FreeDofs(el_int)) * f.vec
s = Draw(mesh, deformation=CoefficientFunction(sgf.components[0]+sgf.components[1]), name="defo_sol")
if s is not None:
    s.setDeformation(True)
    s.setDeformationScale(2)

# quit()

aux_mat = pc_aux.aux_mat
aux_inv = aux_mat.Inverse()
pca = (pc_aux.P @ aux_inv @ pc_aux.P.T)

sgf2 = GridFunction(X)
sgf2.vec.data = pca * f.vec
s2 = Draw(mesh, deformation=CoefficientFunction(sgf2.components[0]+sgf2.components[1]), name="defo_aux")
if s2 is not None:
    s2.setDeformation(True)
    s2.setDeformationScale(2)

lamis = list(EigenValues_Preconditioner(mat=a.mat, pre=pca))
print('lamis ', lamis)
if len(lamis):
    print('max. EV ', lamis[-1])
    print('min. EV ', lamis[0])
    print('cond ', lamis[-1]/lamis[0])


# gf = GridFunction(X.components[1])
# Draw(gf, mesh, 'bf')
# mfs = list(mesh.facets)
# for f in [mfs[0]]:
#     dnums = X.components[1].GetDofNrs(f)
#     print('dnums:', len(dnums))
#     while True:
#         j = int(input('gimme\n'))
#         d = dnums[j]
#         gf.vec[:] = 0
#         gf.vec[d] = 1
#         Redraw()
#         print('dof ' + str(d))
