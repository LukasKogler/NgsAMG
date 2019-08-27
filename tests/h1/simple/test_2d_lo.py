import ngsolve, ngs_amg
from amg_utils import *

def test_2d_lo():
    geo, mesh = gen_square(maxh=0.05, nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_poisson(mesh, order=1, diri="left|top")
    pc_opts = { "ngs_amg_max_coarse_size" : 5,
                "ngs_amg_log_level" : "none",
                "ngs_amg_print_log" : True }
    c = ngsolve.Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms=25)

if __name__ == "__main__":
    test_2d_lo()
