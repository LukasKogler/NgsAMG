import ngsolve, ngs_amg
from amg_utils import *

def test_2d_bddc():
    geo, mesh = gen_square(maxh=0.05, nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_poisson(mesh, order=4, diri="right")
    pc_opts = { "ngs_amg_max_coarse_size" : 5 }
    c = ngsolve.Preconditioner(a, "bddc", coarsetype = "ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms=60)

if __name__ == "__main__":
    test_2d_bddc()
