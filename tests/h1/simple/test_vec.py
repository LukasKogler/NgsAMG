import ngsolve as ns
import ngs_amg, sys
from amg_utils import *

def do_test_vec_h1(mesh_dim = 2, vec_dim = 2, maxh = None, nref = 0, space = "mdim", ms = 100):
    print("--- test vector h1, mesh_dim", mesh_dim, "vec_dim", vec_dim, "space", space)
    sys.stdout.flush()
    if mesh_dim == 2:
        geo, mesh = gen_square(maxh=0.1 if maxh is None else maxh, nref=nref, comm=ngsolve.mpi_world)
    else:
        geo, mesh = gen_cube(maxh=0.2 if maxh is None else maxh, nref=nref, comm=ngsolve.mpi_world)
    if space == "mdim":
        V = ngs.H1(mesh, order = 1, dirichlet = "left|right|top|back", dim=vec_dim)
        u,v = V.TnT()
        gradu = ngs.Grad(u)
        gradv = ngs.Grad(v)
    else: # compound/reordered compound
        if vec_dim == mesh_dim:
            V = ngs.VectorH1(mesh, order = 1, dirichlet = "left|right|top|back")
            if space == "reo":
                V = ngs.comp.Reorder(V)
            u,v = V.TnT()
            gradu = ngs.Grad(u)
            gradv = ngs.Grad(v)
        else:
            Vc = ngs.H1(mesh, order = 1, dirichlet = "left|right|top|back")
            V = ngs.FESpace([Vc for k in range(vec_dim)])
            print("manual compound")
            u, v = V.TnT()
            gradu = ngs.CoefficientFunction(tuple(ngs.Grad(x) for x in u))
            u = ngs.CoefficientFunction(tuple(u))
            gradv = ngs.CoefficientFunction(tuple(ngs.Grad(x) for x in v))
            v = ngs.CoefficientFunction(tuple(v))
            if space == "reo":
                V = ngs.comp.Reorder(V)
    a = ngs.BilinearForm(V)
    a += ngs.InnerProduct(gradu, gradv) * ngs.dx
    f = ngs.LinearForm(V)
    fcf = ngs.CoefficientFunction(tuple([1 for k in range(vec_dim)]))
    f += ngs.InnerProduct(fcf, v) * ngs.dx
    pc_opts = { "ngs_amg_max_coarse_size" : 5,
                "ngs_amg_log_level" : "extra",
                "ngs_amg_print_log" : True }
    if vec_dim == 2:
        c = ngs_amg.h1_2d(a, **pc_opts)
    else:
        c = ngs_amg.h1_3d(a, **pc_opts)
    Solve(a, f, c, ms=ms)
    print("--- DONE with test vector h1, mesh_dim", mesh_dim, "vec_dim", vec_dim, "space", space)
    sys.stdout.flush()

def test_22_mdim():
    do_test_vec_h1(mesh_dim = 2, vec_dim = 2, maxh = 0.05, nref = 0, space = "mdim", ms = 30)
def test_23_mdim():
    do_test_vec_h1(mesh_dim = 2, vec_dim = 3, maxh = 0.05, nref = 0, space = "mdim", ms = 30)
def test_32_mdim():
    do_test_vec_h1(mesh_dim = 3, vec_dim = 2, maxh = 0.1, nref = 0, space = "mdim", ms = 30)
def test_33_mdim():
    do_test_vec_h1(mesh_dim = 3, vec_dim = 3, maxh = 0.1, nref = 0, space = "mdim", ms = 30)

def test_22_comp():
    do_test_vec_h1(mesh_dim = 2, vec_dim = 2, maxh = 0.05, nref = 0, space = "compound", ms = 30)
def test_23_comp():
    do_test_vec_h1(mesh_dim = 2, vec_dim = 3, maxh = 0.05, nref = 0, space = "compound", ms = 30)
def test_32_comp():
    do_test_vec_h1(mesh_dim = 3, vec_dim = 2, maxh = 0.1, nref = 0, space = "compound", ms = 30)
def test_33_comp():
    do_test_vec_h1(mesh_dim = 3, vec_dim = 3, maxh = 0.1, nref = 0, space = "compound", ms = 30)
    
# def test_22_reo():
#     do_test_vec_h1(mesh_dim = 2, vec_dim = 2, maxh = 0.05, nref = 0, space = "reo", ms = 30)
# def test_23_reo():
#     do_test_vec_h1(mesh_dim = 2, vec_dim = 3, maxh = 0.05, nref = 0, space = "reo", ms = 30)
# def test_32_reo():
#     do_test_vec_h1(mesh_dim = 3, vec_dim = 2, maxh = 0.1, nref = 0, space = "reo", ms = 30)
# def test_33_reo():
#     do_test_vec_h1(mesh_dim = 3, vec_dim = 3, maxh = 0.1, nref = 0, space = "reo", ms = 30)

if __name__ == "__main__":
    test_22_mdim()
    test_23_mdim()
    test_32_mdim()
    test_33_mdim()
    test_22_comp()
    test_23_comp()
    test_32_comp()
    test_33_comp()
    # reordered does not work anymore (now ordered by elements)
    # test_22_reo()
    # test_23_reo()
    # test_32_reo()
    # test_33_reo()
    
