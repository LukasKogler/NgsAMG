import ngsolve, ngs_amg
from amg_utils import *

# wirebasket without edge bubbles
def test_3d_bddc_v1():
    geo, mesh = gen_cube(maxh=0.25, nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_poisson(mesh, order=4, fes_opts = { "wb_withedges" : False }, diri="top|right|left")
    pc_opts = { "ngs_amg_max_coarse_size" : 5 }
    c = ngsolve.Preconditioner(a, "bddc", coarsetype="ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms = 200)

# wirebasket with edge bubbles, coarsening on LO DOFs
# note: this is really, really bad, but keep it in to make sure it works
def test_3d_bddc_v2():
    geo, mesh = gen_cube(maxh=0.4, nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_poisson(mesh, order=4, beta=1, diri="")#diri="right|left")
    pc_opts = { "ngs_amg_max_coarse_size" : 5, "ngs_amg_do_test" : True, "ngs_amg_clev" : "none", "ngs_amg_max_levels" : 2, "ngs_amg_oldsm" : True }
    c = ngsolve.Preconditioner(a, "bddc", coarsetype="ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms = 250)

# nodal P2 basis, wirebasket with lowest order edge + vertex DOFs
def test_3d_bddc_nodalp2():
    geo, mesh = gen_cube(maxh=0.25, nref=0, comm=ngsolve.mpi_world)
    V, a, f = setup_poisson(mesh, order=4, fes_opts = {"nodalp2" : True}, diri="top|left")
    pc_opts = { "ngs_amg_on_dofs" : "select",
                "ngs_amg_subset" : "free",
                "ngs_amg_max_coarse_size" : 5 }
    c = ngsolve.Preconditioner(a, "bddc", coarsetype = "ngs_amg.h1_scal", **pc_opts)
    Solve(a, f, c, ms = 100)

if __name__ == "__main__":
    test_3d_bddc_v1()
    test_3d_bddc_v2()
    test_3d_bddc_nodalp2()
    
