import ngsolve, ngs_amg, sys
from amg_utils import *

def do_test (rots, reo, ms=50):
    print('======= test 2d, lo, rots =', rots, ', reorder=', reo)
    sys.stdout.flush()
    geo, mesh = gen_beam(dim=2, maxh=0.2, lens=[6,1], nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, order=3, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005) ), diri="left", multidim = False, reorder = reo)
    ngsolve.ngsglobals.msg_level = 5
    pc_opts = { "ngs_amg_max_coarse_size" : 10 }
    if reo == "sep":
        pc_opts["ngs_amg_dof_blocks"] = [2,1] if rots else [2]
    elif reo is not False:
        pc_opts["ngs_amg_dof_blocks"] = [3] if rots else [2]
    if rots:
        pc_opts["ngs_amg_rots"] = True
    c = ngsolve.Preconditioner(a, "ngs_amg.elast_2d", **pc_opts)
    Solve(a, f, c, ms=ms)
    print('======= completed test 2d, lo, rots =', rots, ', reorder=', reo, '\n\n')
    sys.stdout.flush()

def test_2d_ho():
    do_test(False, False)

def test_2d_ho_R():
    do_test(True, False)

# # does not work - I cannot find out the low order DOFs!
# def test_2d_lo_ro():    
#     do_test(False, True)
# def test_2d_lo_ro_R():
#     do_test(True, True)
# def test_2d_lo_ro2_R():
#     do_test(True, "sep")

if __name__ == "__main__":
    test_2d_ho()
    # test_2d_ho_ro()
    test_2d_ho_R()
    # test_2d_ho_ro_R()
    # test_2d_ho_ro2_R()
