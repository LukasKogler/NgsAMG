import ngsolve, ngs_amg
from amg_utils import *

def test_3d_lo():
    geo, mesh = gen_beam(dim=3, maxh=0.25, lens=[10,1,1], nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, rotations = False, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0.002) ), diri="left")
    pc_opts = { "ngs_amg_reg_rmats" : True,
                "ngs_amg_max_coarse_size" : 10 }
    c = ngsolve.Preconditioner(a, "ngs_amg.elast3d", **pc_opts)
    Solve(a, f, c, ms=40)

def test_3d_lo_R():
    geo, mesh = gen_beam(dim=3, maxh=0.25, lens=[2,1,1], nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, rotations = True, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0.002) ), diri="left")
    pc_opts = { "ngs_amg_rots" : True,
                "ngs_amg_max_coarse_size" : 10 }
    c = ngsolve.Preconditioner(a, "ngs_amg.elast3d", **pc_opts)
    Solve(a, f, c, ms=40)

if __name__ == "__main__":
    test_3d_lo()
    test_3d_lo_R()
