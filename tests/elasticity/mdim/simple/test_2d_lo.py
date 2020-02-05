import ngsolve, ngs_amg
from amg_utils import *

def test_2d_lo():
    geo, mesh = gen_beam(dim=2, maxh=0.1, lens=[10,1], nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, rotations = False, f_vol = ngsolve.CoefficientFunction( (0, -0.005) ), diri="left")
    print('V ndof', V.ndof)
    pc_opts = { "ngs_amg_max_coarse_size" : 20 }
    c = ngsolve.Preconditioner(a, "ngs_amg.elast_2d", **pc_opts)
    Solve(a, f, c, ms=50)
    
def test_2d_lo_R():
    geo, mesh = gen_beam(dim=2, maxh=0.1, lens=[10,1], nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, rotations = True, f_vol = ngsolve.CoefficientFunction( (0, -0.005) ), diri="left")
    pc_opts = { "ngs_amg_rots" : True,
                "ngs_amg_max_coarse_size" : 20 }
    c = ngsolve.Preconditioner(a, "ngs_amg.elast_2d", **pc_opts)
    Solve(a, f, c, ms=50)

if __name__ == "__main__":
    test_2d_lo()
    test_2d_lo_R()
