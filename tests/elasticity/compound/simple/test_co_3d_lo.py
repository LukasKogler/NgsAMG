import ngsolve, ngs_amg, sys
from amg_utils import *

def do_test (rots, reo, ms=50):
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
    c = ngsolve.Preconditioner(a, "ngs_amg.elast_3d", **pc_opts)
    Solve(a, f, c, ms=ms)
    print('======= completed test 3d, lo, rots =', rots, ', reorder=', reo, '\n\n')
    sys.stdout.flush()

def test_3d_lo():
    do_test (False, False)

def test_3d_lo_R():
    do_test (True, False)

# def test_3d_ro_lo():
#     do_test (False, True)

# def test_3d_ro_lo_R():
#     do_test (True, True)

# def test_3d_ro2_lo_R():
#     do_test (True, "sep")
    
if __name__ == "__main__":
    test_3d_lo()
    test_3d_lo_R()
    # test_3d_ro_lo()
    # test_3d_ro_lo_R()
    # test_3d_ro2_lo_R()
