import ngsolve, ngs_amg, sys
from amg_utils import *

# reorder + ho-smooth does not work - try bddc

def do_test (rots=False, nodalp2 = False, ms=100, order=3, reo = False, use_bddc = False, ho_wb = False):
    print('======= test 3d, ho, rots =', rots, ', reorder=', reo, ', nodalp2=', nodalp2, ', use_bddc=', use_bddc, ', order=', order, 'ho_wb=', ho_wb, "ms=", ms)
    # if reo is not False:
    #     if (order==2) and (nodalp2 is False):
    #         raise "order 2 + reorder only with nodalp2 (ho smoothing + reorder does now work)"
    #     elif order>=3:
    #         raise "ho smoothing does not work with reorder!"
    # if use_bddc and order==2 and nodalp2:
    #     raise "bddc makes no sense here!"
    if not rots and reo=="sep": # "sep" and True are the same here
        return
    sys.stdout.flush()
    geo, mesh = gen_beam(dim=3, maxh=0.4, lens=[4,1,1], nref=0, comm=ngsolve.mpi_world)
    # geo, mesh = gen_beam(dim=3, maxh=0.3, lens=[5,1,1], nref=0, comm=ngsolve.mpi_world)
    fes_opts = dict()
    if nodalp2:
        if order>=2:
            fes_opts["nodalp2"] = True
    elif use_bddc and not ho_wb: # do not take edge-bubbles to crs space
        fes_opts["wb_withedges"] = False
    V, a, f = setup_elast(mesh, order=order, fes_opts = fes_opts, rotations = rots, f_vol = ngsolve.CoefficientFunction( (0, -0.005, 0) ), diri="left", multidim = False, reorder = reo)
    ngsolve.ngsglobals.msg_level = 5
    pc_opts = { "ngs_amg_max_coarse_size" : 15,
                "ngs_amg_do_test" : True,
                "ngs_amg_agg_wt_geom" : False }
    # if nodalp2:
        # pc_opts["ngs_amg_crs_alg"] = "ecol"
    # rotations or no rotataions?
    if rots:
        pc_opts["ngs_amg_rots"] = True
    # subset of the low order space    
    if use_bddc:
        if nodalp2 or order==1 or not ho_wb:
            pc_opts["ngs_amg_on_dofs"] = "select"
            pc_opts["ngs_amg_subset"] = "free"
        else:
            # pc_opts["ngs_amg_on_dofs"] = "range"   # per default !
            # pc_opts["ngs_amg_lo"] = True   # per default !
            pass
    elif nodalp2:
        if order==2:
            pc_opts["ngs_amg_lo"] = False
        else:
            pc_opts["ngs_amg_on_dofs"] = "select"
            pc_opts["ngs_amg_subset"] = "nodalp2"
    # ordering of DOFs within subset
    if reo is not False:
        if not rots:
            pc_opts["ngs_amg_dof_blocks"] = [3]
        elif reo == "sep":
            pc_opts["ngs_amg_dof_blocks"] = [3,3]
        else:
            pc_opts["ngs_amg_dof_blocks"] = [6]
    if order >= 2 and nodalp2:
        pc_opts["ngs_amg_edge_thresh"] = 0.02

    print("pc_opts: ", pc_opts)
    sys.stdout.flush()
    if use_bddc:
        c = ngsolve.Preconditioner(a, "bddc", coarsetype="ngs_amg.elast_3d", **pc_opts)
    else:
        c = ngsolve.Preconditioner(a, "ngs_amg.elast_3d", **pc_opts)
    Solve(a, f, c, ms=ms)
    print('======= completed test 3d, ho, rots =', rots, ', reorder=', reo, ', nodalp2=', nodalp2, ', use_bddc=', use_bddc, ', order=', order, 'ho_wb=', ho_wb, "ms=", ms)
    sys.stdout.flush()

# coarsen on p1 DOFs, smooth on others [no reorder!]
def test_3d_ho():
    for R in [True, False]:
        for reo in [False]:#, True, "sep"]:
            do_test(rots=R)

# coarsen on p2 DOFs (so all DOFs) 
def test_3d_np2():
    for R in [True, False]:
        for reo in [False]:#, True, "sep"]:
            do_test(rots=R, order=2, nodalp2=True, reo=reo)

# coarsen on p2 DOFs, smooth on HO [no reorder]
def test_3d_np2_ho():
    for R in [True, False]:
        for reo in [False]:#, True, "sep"]:
            do_test(rots=R, order=3, nodalp2=True)

# BDDC reduces to p1 DOFs, coarsening on all left
def test_3d_bddc():
    for R in [True, False]:
        for reo in [False]:#, True, "sep"]:
            do_test(rots=R, order=3, nodalp2=False, use_bddc=True, reo=reo, ho_wb=False, ms=200)

# BDDc reduces to p1+p2 DOFs, coarsening on all left
def test_3d_bddc_np2():
    for R in [True, False]:
        for reo in [False]:#, True, "sep"]:
            do_test(rots=R, order=3, nodalp2=True, use_bddc=True, reo=reo)

# BDDc reduces to p1 DOFs + edge bubbles, coarsening only on P1 dofs [no reorder!]
def test_3d_bddc_howb():
    for R in [True, False]:
        do_test(rots=R, order=3, nodalp2=False, use_bddc=True, reo=False, ho_wb=True)


if __name__ == "__main__":
    test_3d_ho()
    test_3d_np2()
    test_3d_np2_ho()
    test_3d_bddc()
    test_3d_bddc_np2()
    test_3d_bddc_howb()
