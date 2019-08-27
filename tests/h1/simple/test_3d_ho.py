import ngsolve, ngs_amg
from amg_utils import *

def test_3d_ho():
    geo, mesh = gen_cube(maxh=0.3, nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_poisson(mesh, order=5, diri="right|top")
    pc_opts = { "ngs_amg_max_coarse_size" : 5 }
    c = ngsolve.Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms = 250)

def test_3d_nodalp2():
    geo, mesh = gen_cube(maxh=0.3, nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_poisson(mesh, order=2, fes_opts = {"nodalp2" : True}, diri="left|bot")
    pc_opts = { "ngs_amg_lo" : False,
                "ngs_amg_max_coarse_size" : 5 }
    c = ngsolve.Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms = 50)

def test_3d_ho_nodalp2():
    geo, mesh = gen_cube(maxh=0.3, nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_poisson(mesh, order=5, fes_opts = {"nodalp2" : True}, diri="left|right")
    pc_opts = { "ngs_amg_on_dofs" : "select",
                "ngs_amg_subset" : "nodalp2",
                "ngs_amg_max_coarse_size" : 5 }
    c = ngsolve.Preconditioner(a, "ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms = 280)


if __name__ == "__main__":
    test_3d_ho()
    test_3d_nodalp2()
    test_3d_ho_nodalp2()
