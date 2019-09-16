import ngsolve, ngs_amg, sys
from amg_utils import *

def do_test_2d_lo(jump):
    print('test sq_with_sqs, jump =', jump)
    sys.stdout.flush()
    geo, mesh = gen_sq_with_sqs(maxh=0.05, nref=1, comm=ngsolve.mpi_world)
    a0 = 1
    alpha = { "mat_a" : a0, "mat_b" : jump * a0 }
    V, a, f = setup_poisson(mesh, order=1, diri="outer_.*", alpha = ngsolve.CoefficientFunction([alpha[name] for name in mesh.GetMaterials()]) )
    pc_opts = { "ngs_amg_max_coarse_size" : 5,
                "ngs_amg_log_level" : "none",
                "ngs_amg_print_log" : True }
    c = ngsolve.Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms=25, tol=1e-6)

def do_test_2d_lo_fiber(jump):
    print('test 2d fibers, jump =', jump)
    sys.stdout.flush()
    geo, mesh = gen_fibers_2d(maxh=0.1, nref=1, comm=ngsolve.mpi_world)
    a0 = 1
    alpha = { "mat_a" : a0, "mat_b" : jump * a0 }
    V, a, f = setup_poisson(mesh, order=1, diri="outer_top|outer_bot", alpha = ngsolve.CoefficientFunction([alpha[name] for name in mesh.GetMaterials()]) )
    pc_opts = { "ngs_amg_max_coarse_size" : 5,
                "ngs_amg_log_level" : "none",
                "ngs_amg_print_log" : True }
    c = ngsolve.Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms=25, tol=1e-6)

def test_2d_lo_1():
    do_test_2d_lo(1e1)

def test_2d_lo_2():
    do_test_2d_lo(1e2)

def test_2d_lo_4():
    do_test_2d_lo(1e4)

def test_2d_lo_6():
    do_test_2d_lo(1e6)

def test_2d_lo_fiber_1():
    do_test_2d_lo_fiber(1e1)

def test_2d_lo_fiber_2():
    do_test_2d_lo_fiber(1e2)

def test_2d_lo_fiber_4():
    do_test_2d_lo_fiber(1e4)

def test_2d_lo_fiber_6():
    do_test_2d_lo_fiber(1e6)

    
if __name__ == "__main__":
    test_2d_lo_1()
    test_2d_lo_2()
    test_2d_lo_4()
    test_2d_lo_6()
    test_2d_lo_fiber_1()
    test_2d_lo_fiber_2()
    test_2d_lo_fiber_4()
    test_2d_lo_fiber_6()
