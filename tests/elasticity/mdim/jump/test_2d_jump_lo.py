import ngsolve, ngs_amg
from amg_utils import *

def do_test_2d_lo(jump, rots, ms=40):
    geo, mesh = gen_sq_with_sqs(maxh=0.1, nref=0, comm=ngsolve.mpi_world)
    a0 = 1
    mu = { "mat_a" : a0, "mat_b" : jump * a0 }
    V, a, f = setup_elast(mesh, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005) ), diri="outer_left",
                          mu = ngsolve.CoefficientFunction([mu[name] for name in mesh.GetMaterials()]) )
    print('V ndof', V.ndof)
    pc_opts = { "ngs_amg_max_coarse_size" : 10,
                "ngs_amg_log_level" : "extra",
                "ngs_amg_print_log" : True }
    if rots:
        pc_opts["ngs_amg_rots"] = True
    c = ngsolve.Preconditioner(a, "ngs_amg.elast_2d", **pc_opts)
    Solve(a, f, c, ms=50)

def test_2d_lo_1():
    do_test_2d_lo(1e1, False)
    
def test_2d_lo_2():
    do_test_2d_lo(1e2, False)

def test_2d_lo_4():
    do_test_2d_lo(1e4, False)

def test_2d_lo_6():
    do_test_2d_lo(1e6, False)

def test_2d_lo_R_1():
    do_test_2d_lo(1e1, True)

def test_2d_lo_R_2():
    do_test_2d_lo(1e2, True)

def test_2d_lo_R_4():
    do_test_2d_lo(1e4, True)

def test_2d_lo_R_6():
    do_test_2d_lo(1e6, True)

if __name__ == "__main__":
    # 2d, lo, no rots
    do_test_2d_lo(1e1, False)
    do_test_2d_lo(1e2, False)
    do_test_2d_lo(1e4, False)
    do_test_2d_lo(1e6, False)
    ## 2d, lo, rots
    do_test_2d_lo(1e1, True)
    do_test_2d_lo(1e2, True)
    do_test_2d_lo(1e4, True)
    do_test_2d_lo(1e6, True)
