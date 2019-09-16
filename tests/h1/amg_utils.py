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


def gen_sq_with_sqs(maxh, nref, comm):
    geo = netgen.geom2d.SplineGeometry()

    geo.AddRectangle( (-1,-1), (1,1), bcs=["outer_bot", "outer_right", "outer_top", "outer_left"], leftdomain=1, rightdomain=0)


    geo.AddRectangle( (-0.6,0.4), (-0.4,0.6), bc="inner", leftdomain=2, rightdomain=1)
    geo.AddRectangle( (0.4,0.4), (0.6,0.6), bc="inner", leftdomain=2, rightdomain=1)

    geo.AddRectangle( (-0.15,-0.15), (0.15,0.15), bc="inner", leftdomain=2, rightdomain=1)

    geo.AddRectangle( (-0.8,-0.6), (0.8,-0.4), bc="inner", leftdomain=2, rightdomain=1)

    # really close to the central square
    geo.AddRectangle( (0.2,-0.1), (0.4,0.1), bc="inner", leftdomain=2, rightdomain=1)

    geo.SetMaterial(1, "mat_a")
    geo.SetMaterial(2, "mat_b")

    return gen_ref_mesh(geo, maxh, nref, comm)

def MyMakeRectangle (geo, p1, p2, bc=None, bcs=None, rightdomain=[0,0,0,0], **args):
    p1x, p1y = p1
    p2x, p2y = p2
    p1x,p2x = min(p1x,p2x), max(p1x, p2x)
    p1y,p2y = min(p1y,p2y), max(p1y, p2y)
    if not bcs: bcs=4*[bc]
    pts = [geo.AppendPoint(*p) for p in [(p1x,p1y), (p2x, p1y), (p2x, p2y), (p1x, p2y)]]
    for p1,p2,bc,rd in [(0,1,bcs[0], rightdomain[0]), (1, 2, bcs[1], rightdomain[1]), (2, 3, bcs[2], rightdomain[2]), (3, 0, bcs[3], rightdomain[3])]:
        geo.Append( ["line", pts[p1], pts[p2]], bc=bc, rightdomain=rd, **args)


def make_fiber_geo(n_fibers = 3, fiber_box = [10,1], fiber_rad = 0.25, maxh_mat = 0.5, maxh_fiber = 0.3):
    geo = netgen.geom2d.SplineGeometry()
    geo.SetMaterial(1, "mat_a")
    geo.SetMaterial(2, "mat_b")

    xmax = fiber_box[0]

    # Points
    y = n_fibers * fiber_box[1]
    geo.AppendPoint(0, y)
    geo.AppendPoint(xmax, y)
    # top points:
    for n in range(n_fibers):
        #top fiber
        y = y - (0.5*fiber_box[1]-fiber_rad)
        geo.AppendPoint(0, y)
        geo.AppendPoint(xmax, y)
        y = y - 2*fiber_rad
        #bot fiber
        geo.AppendPoint(0, y)
        geo.AppendPoint(xmax, y)
        #top next fiber
        y = y - (0.5*fiber_box[1]-fiber_rad)
    #bot box
    geo.AppendPoint(0, y)
    geo.AppendPoint(xmax, y)

    #fibers, between, top+bot
    n_layers = n_fibers + (n_fibers-1) + 2
    n_points = 2 * (n_layers+1)

    # Segments
    # top seg
    geo.Append( ["line", 0, 1], bc="outer_top", leftdomain=0, rightdomain=1)
    # bot seg
    geo.Append( ["line", n_points-2, n_points-1], bc="outer_bot", leftdomain=1, rightdomain=0)
    # interface segs
    ld = 1
    for k in range(n_layers-1):
        geo.Append( ["line", 2+2*k, 2+2*k+1], bc="inner", leftdomain=ld, rightdomain=3-ld)
        ld = 3-ld
        
    # left/right segs
    ind = 1
    for k in range(n_layers):
        geo.Append( ["line", 2*k, 2*(k+1)], bc="outer_left", leftdomain=ind, rightdomain=0)
        geo.Append( ["line", 1+2*k, 1+2*(k+1)], bc="outer_right", leftdomain=0, rightdomain=ind)
        ind = 3-ind
        
    return geo

def gen_fibers_2d(maxh, nref, comm):
    geo = make_fiber_geo(8, fiber_box = [6,0.4], fiber_rad = 0.1, maxh_mat = maxh, maxh_fiber = maxh)
    return gen_ref_mesh(geo, maxh, nref, comm)

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

def Solve(a, f, c, ms = 100, tol=1e-12, nocb=True):
    ngs.ngsglobals.msg_level = 5
    gfu = ngs.GridFunction(a.space)
    with ngs.TaskManager():
        a.Assemble()
        f.Assemble()
        ngs.ngsglobals.msg_level = 1
        c.Test()
        cb = None if a.space.mesh.comm.rank != 0 or nocb else lambda k, x: print("it =", k , ", err =", x)
        cg = ngs.krylovspace.CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=ms, tol=tol)
        ngs.mpi_world.Barrier()
        ts = ngs.mpi_world.WTime()
        cg.Solve(sol=gfu.vec, rhs=f.vec)
        ngs.mpi_world.Barrier()
        ts = ngs.mpi_world.WTime() - ts
        if ngs.mpi_world.rank == 0:
            print("---")
            print("multi-dim ", a.space.dim)
            print("(vectorial) ndof ", a.space.ndofglobal)
            print("(scalar) ndof ", a.space.dim * a.space.ndofglobal)
            print("used nits = ", cg.iterations)
            print("(vec) dofs / (sec * np) = ", a.space.ndofglobal / (ts * max(ngs.mpi_world.size-1,1)))
            print("(scal) dofs / (sec * np) = ", (a.space.ndofglobal * a.space.dim) / (ts * max(ngs.mpi_world.size-1,1)))
            print("---")
    assert cg.errors[-1] < tol * cg.errors[0]
    assert cg.iterations < ms
    return gfu


