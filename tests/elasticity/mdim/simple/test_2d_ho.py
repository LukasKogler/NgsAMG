import ngsolve, ngs_amg
from amg_utils import *


def do_ho_smooth(rots, ms, pc_opts):
    print('test 2d, rots  = ', rots)
    geo, mesh = gen_beam(dim = 2, maxh = 0.2, lens = [10,1], nref = 0, comm = ngsolve.mpi_world)
    V, a, f = setup_elast(mesh, order = 4, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005) ), diri = "left")
    print('V ndof', V.ndof)
    pc_opts["ngs_amg_rots"]  =  rots
    c  =  ngsolve.Preconditioner(a, "ngs_amg.elast_2d", **pc_opts)
    Solve(a, f, c, ms = ms)

def test_2d_ho():
    do_ho_smooth(False, 30, { "ngs_amg_max_coarse_size" : 10 })

def test_2d_ho_R():
    do_ho_smooth(True, 85, { "ngs_amg_max_coarse_size" : 10 })

    
if __name__ == "__main__":
    test_2d_ho()
    test_2d_ho_R()
    # note: nodalp2 does nothing for p2
