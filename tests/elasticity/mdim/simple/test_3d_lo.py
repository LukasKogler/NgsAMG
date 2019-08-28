import ngsolve, ngs_amg
from amg_utils import *

def test_3d_lo():
    geo, mesh = gen_beam(dim=3, maxh=0.1, lens=[10,1,1], nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, rotations = False, f_vol = CoefficientFunction( (0, -0.005) ), diri="left")
    pc_opts = {}
    c = ngsolve.Preconditioner(a, "ngs_amg.elast3d", **pc_opts)
    Solve(a, f, c, ms=25)

def test_3d_lo_R():
    geo, mesh = gen_beam(dim=3, maxh=0.1, lens=[10,1,1], nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, rotations = True, f_vol = CoefficientFunction( (0, -0.005) ), diri="left")
    pc_opts = {}
    c = ngsolve.Preconditioner(a, "ngs_amg.elast3d", **pc_opts)
    Solve(a, f, c, ms=25)

if __name__ == "__main__":
    test_3d_lo()
    test_3d_lo_R()
