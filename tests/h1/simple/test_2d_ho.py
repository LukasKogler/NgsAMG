import ngsolve, ngs_amg
from amg_utils import *

def test_2d_ho():
    geo, mesh = gen_square(maxh=0.05, nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_poisson(mesh, order=3, diri="left|top")
    pc_opts = { "ngs_amg_max_coarse_size" : 5 }
    c = ngsolve.Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms=40)

if __name__ == "__main__":
    test_2d_ho()
