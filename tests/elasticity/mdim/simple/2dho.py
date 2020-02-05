import ngsolve, ngs_amg
from amg_utils import *

rots = True
order = 4
ms = 3
maxh = 0.35
pc_opts = { "ngs_amg_max_coarse_size" : 10 }
dim = 2
noinv = False

print('test 2d with nodalp2, order  = ', order, 'rots  = ', rots)

if dim == 2:
    geo, mesh = gen_beam(dim = 2, maxh = maxh, lens = [1,1], nref = 0, comm = ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, order = order, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005) ), diri = "left")
else:
    geo, mesh = gen_beam(dim = 3, maxh = maxh, lens = [1,1,1], nref = 0, comm = ngsolve.mpi_world)
    fo = dict()
#    if order > 1:
#        fo["nodalp2"] = True
    V, a, f = setup_elast(mesh, order = order, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0) ), diri = "left",
                          fes_opts = fo)

hofrees = []
hofrees_ex = []
hofrees_loc = []

print('ndofs are ', V.lospace.ndof, V.ndof)
import sys
sys.stdout.flush()

if ngsolve.mpi_world.size > 10:
    pds = V.ParallelDofs()

    for k in range(V.lospace.ndof, V.ndof):
        if V.FreeDofs()[k] == True:
            hofrees.append(k)
            if len(pds.Dof2Proc(k)):
                hofrees_ex.append(k)
            else:
                hofrees_loc.append(k)
        # V.FreeDofs()[k] = False
        #    V.SetCouplingType(k, COUPLING_TYPE.HIDDEN_DOF)

    for k in range(V.lospace.ndof):
        V.FreeDofs()[k] = False

    if ngsolve.mpi_world.rank == 1:
        extra_diri = [2,6,7,9]
    elif ngsolve.mpi_world.rank == 2:
        extra_diri = [5,6,8,9]
    else:
        extra_diri = []

    # for dof in extra_diri:
    #     V.FreeDofs()[dof] = False

    for k in range(V.ndof):
        if len(pds.Dof2Proc(k)) != 0:
                V.FreeDofs()[k] = False

    if ngsolve.mpi_world.rank != 0:
        print('set free ', extra_diri[0])
        V.FreeDofs()[extra_diri[2]] = True
        pass

    # if ngsolve.mpi_world.rank == 2:
    #     for k in range(V.ndof):
    #         if len(pds.Dof2Proc(k)) == 0:
    #             V.FreeDofs()[k] = False

        
    print('free / ex: ')
    for k in range(V.ndof):
        print(k, V.FreeDofs()[k], len(pds.Dof2Proc(k))>0)

    if ngsolve.mpi_world.rank == 1:
        hfl = len(hofrees_loc)
        print('hfl:', hfl)
        print('set free:', hofrees_loc[6])
        print('hofrees_loc:', hofrees_loc)
        print('hofrees_ex:', hofrees_ex)
        V.FreeDofs()[10 + 15] = True
        #for k in hofrees_loc[0:1]:
        #    V.FreeDofs()[k] = True

    import sys
    sys.stdout.flush()

    W = ngsolve.H1(mesh, order=order)
    gfbf = ngsolve.GridFunction(W)
    gfbf.vec[:] = 0
    if ngsolve.mpi_world.rank == 1:
        gfbf.vec[10 + 15] = 1

    ngsolve.Draw(mesh, name="mesh")
    ngsolve.Draw(gfbf, mesh, name="bf")

dobft = False
if dobft:
    while True:
        bf_nr = int(0)
        if ngsolve.mpi_world.rank == 0:
            bf_nr = int(input('nr ?'))
            print('got nr', bf_nr)
        bf_nr = ngsolve.mpi_world.Sum(bf_nr)
        if ngsolve.mpi_world.rank == 1:
            gfbf.vec[:] = 0
            gfbf.vec[bf_nr] = 1
        ngsolve.mpi_world.Barrier()
        ngsolve.Redraw()
    
ngsolve.ngsglobals.msg_level = 5
print('V ndof', V.ndof)
pc_opts["ngs_amg_rots"] = rots
pc_opts["ngs_amg_reg_mats"] = False
pc_opts["ngs_amg_reg_rmats"] = False
pc_opts["ngs_amg_max_coarse_size"] = 3
pc_opts["ngs_amg_max_levels"] = 2
pc_opts["ngs_amg_first_aaf"] = 0.4
if noinv == True:
    pc_opts["ngs_amg_clev"] = "none"
pc_opts["ngs_amg_cinv_type"] = "masterinverse"
pc_opts["ngs_amg_sp_omega"] = 1.0
pc_opts["ngs_amg_log_level"] = "extra"
pc_opts["ngs_amg_enable_redist"] = True
pc_opts["ngs_amg_enable_sp"] = False
pc_opts["ngs_amg_sm_ver"] = 3
if order  ==  2:
    pass
    # pc_opts["ngs_amg_lo"] = False
    # pc_opts["ngs_amg_force_nolo"] = True
else:
    pc_opts["ngs_amg_on_dofs"] = "subset"
    pc_opts["ngs_amg_subset"] = "select"
if dim==2:
    # c = ngsolve.Preconditioner(a, "ngs_amg.elast2d", **pc_opts)
    c = ngs_amg.elast_2d(a, **pc_opts)
else:
    # c = ngsolve.Preconditioner(a, "ngs_amg.elast3d", **pc_opts)
    c = ngs_amg.elast_3d(a, **pc_opts)

#a.Assemble()
#f.Assemble()
ngsolve.ngsglobals.msg_level = 1

gfu = Solve(a, f, c, ms = 3)

c.Test()

dodraw = False
if dodraw:
    # gfu = ngsolve.GridFunction(V)
    # gfu.vec.data = c.mat * f.vec

    gfu2 = ngsolve.GridFunction(V)
    gfu2.vec.data = a.mat.Inverse(V.FreeDofs()) * f.vec

    gfu3 = ngsolve.GridFunction(V)
    gfu3.vec.data = gfu2.vec - gfu.vec

    ngsolve.Draw(mesh, deformation=ngsolve.CoefficientFunction(tuple((gfu[0], gfu[1]))), name="disp")
    ngsolve.Draw(mesh, deformation=ngsolve.CoefficientFunction(tuple((gfu2[0], gfu2[1]))), name="ex_disp")
    ngsolve.Draw(mesh, deformation=ngsolve.CoefficientFunction(tuple((gfu3[0], gfu3[1]))), name="diff_disp")

    ngsolve.Draw(gfu[2], mesh, name="rot")
    ngsolve.Draw(gfu2[2], mesh, name="ex_rot")
    ngsolve.Draw(gfu3[2], mesh, name="diff_rot")


    ngsolve.Draw(gfu, mesh, name="all")
    ngsolve.Draw(gfu2, mesh, name="all_ex")
    ngsolve.Draw(gfu3, mesh, name="all_diff")



#from bftester_vec import *
#shape_test(mesh, maxh, V, a, c, 3, order=order)
