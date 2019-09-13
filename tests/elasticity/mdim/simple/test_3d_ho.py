import sys, ngsolve, ngs_amg
from amg_utils import *


def do_ho_smooth(rots, ms, pc_opts):
    print('test 3d, rots  = ', rots)
    sys.stdout.flush()
    geo, mesh = gen_beam(dim = 3, maxh = 0.35, lens = [5,1,1], nref = 0, comm = ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, order = 2, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0) ), diri = "left")
    pc_opts["ngs_amg_rots"]  =  rots
    if not rots:
        pc_opts["ngs_amg_reg_mats"] = True
        pc_opts["ngs_amg_reg_rmats"] = True
    c  =  ngs_amg.elast_3d(a, **pc_opts)
    Solve(a, f, c, ms = ms)

def test_3d_ho():
    do_ho_smooth(False, 30, { "ngs_amg_max_coarse_size" : 10 })

def test_3d_ho_R():
    do_ho_smooth(True, 60 , { "ngs_amg_max_coarse_size" : 10 })


def do_ho_smooth_nodalp2(rots, ms, pc_opts, order = 3):
    print('test 3d with nodalp2, order  = ', order, 'rots  = ', rots)
    sys.stdout.flush()
    geo, mesh = gen_beam(dim = 3, maxh = 0.35, lens = [5,1,1], nref = 0, comm = ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, order = order, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0) ), diri = "left",
                          fes_opts = {"nodalp2" : True} )
    print('V ndof', V.ndof)
    pc_opts["ngs_amg_rots"] = rots
    pc_opts["ngs_amg_first_aaf"] = 0.1
    if order  ==  2:
        pc_opts["ngs_amg_lo"] = False
    else:
        pc_opts["ngs_amg_on_dofs"] = "select"
        pc_opts["ngs_amg_subset"] = "nodalp2"
    if rots is False:
        pc_opts["ngs_amg_reg_mats"] = True
        pc_opts["ngs_amg_reg_rmats"] = True
    c = ngs_amg.elast_3d(a, **pc_opts)
    Solve(a, f, c, ms = ms)


def test_3d_np2():
    do_ho_smooth_nodalp2(False, 30, { "ngs_amg_max_coarse_size" : 10 }, 2)
    
def test_3d_np2_R():
    do_ho_smooth_nodalp2(True, 30, { "ngs_amg_max_coarse_size" : 10 }, 2)

def test_3d_np2_ho():
    do_ho_smooth_nodalp2(False, 30, { "ngs_amg_max_coarse_size" : 10 })
    
def test_3d_np2_ho_R():
    do_ho_smooth_nodalp2(True, 60, { "ngs_amg_max_coarse_size" : 10 })

    
if __name__ == "__main__":
    test_3d_ho()
    test_3d_ho_R()
    test_3d_np2()
    test_3d_np2_R()
    test_3d_np2_ho()
    test_3d_np2_ho_R()
