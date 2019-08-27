import ngsolve as ngs
import netgen, netgen.geom2d, netgen.csg
import ngsolve.krylovspace

def gen_ref_mesh (geo, maxh, nref, comm, mesh_file = '', save = False):
    ngs.ngsglobals.msg_level = 0
    if comm.rank==0:
        if mesh_file == '' or save:
            ngm = geo.GenerateMesh(maxh=maxh)
            if len(mesh_file) and save:
                ngm.Save(mesh_file)
        else:
            ngm = netgen.meshing.Mesh(dim=3)
            ngm.Load(mesh_file)
        if comm.size > 0:
            ngm.Distribute(comm)
    else:
        ngm = netgen.meshing.Mesh.Receive(comm)
        ngm.SetGeometry(geo)
    for l in range(nref):
        ngm.Refine()
    return geo, ngs.comp.Mesh(ngm)
    

def gen_square(maxh, nref, comm):
    return gen_ref_mesh(netgen.geom2d.unit_square, maxh, nref, comm)

def gen_cube(maxh, nref, comm):
    return gen_ref_mesh(netgen.csg.unit_cube, maxh, nref, comm)

def setup_poisson(mesh, alpha=1 , beta=0, f=1, diri=".*", order=1, fes_opts=dict(), blf_opts=dict(), lf_opts=dict()):
    V = ngs.H1(mesh, order=order, dirichlet=diri, **fes_opts)
    u,v = V.TnT()
    a = ngs.BilinearForm(V, **blf_opts)
    a += ngs.SymbolicBFI(alpha * ngs.grad(u)*ngs.grad(v))
    if beta != 0:
        a += ngs.SymbolicBFI(beta * u*v)
    lf = ngs.LinearForm(V)
    lf += ngs.SymbolicLFI(f*v)
    return V, a, lf

def Solve(a, f, c, ms = 100):
    ngs.ngsglobals.msg_level = 0
    gfu = ngs.GridFunction(a.space)
    with ngs.TaskManager():
        a.Assemble()
        f.Assemble()
        ngs.ngsglobals.msg_level = 1
        c.Test()
        cb = None if a.space.mesh.comm.rank != 0 else lambda k, x: print("it =", k , ", err =", x)
        cg = ngs.krylovspace.CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=ms, tol=1e-12)
        cg.Solve(sol=gfu.vec, rhs=f.vec)
        print("used nits = ", cg.iterations)
    assert cg.errors[-1] < 1e-12 * cg.errors[0]
    assert cg.iterations < ms
