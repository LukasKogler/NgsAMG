import ngsolve as ngs
import netgen as ng
import netgen.geom2d as geom2d
import netgen.csg as csg
import ngsolve.krylovspace


def gen_ref_mesh (geo, maxh, nref, comm, mesh_file = '', save = False):
    ngs.ngsglobals.msg_level = 0
    if comm.rank==0:
        if mesh_file == '' or save:
            ngm = geo.GenerateMesh(maxh=maxh)
            if len(mesh_file) and save:
                ngm.Save(mesh_file)
        else:
            ngm = ng.meshing.Mesh(dim=3)
            ngm.Load(mesh_file)
        if comm.size > 0:
            ngm.Distribute(comm)
    else:
        ngm = ng.meshing.Mesh.Receive(comm)
        ngm.SetGeometry(geo)
    for l in range(nref):
        ngm.Refine()
    return geo, ngs.comp.Mesh(ngm)

def distribute_mesh(ngmesh, comm, geo=None):
    if comm.rank==0:
        if comm.size > 0:
            ngmesh.Distribute(comm)
    else:
        ngmesh = ng.meshing.Mesh.Receive(comm)
        if geo is not None:
            ngmesh.SetGeometry(geo)
    return ngs.comp.Mesh(ngmesh)


def gen_2dbeam(maxh, nref, comm, lens = [10,1]):
    geo = geom2d.SplineGeometry()
    geo.AddRectangle((0, 0), (lens[0], lens[1]), leftdomain=1, rightdomain=0, bcs = ("bottom", "right", "top", "left"))
    return gen_ref_mesh(geo, maxh, nref, comm)

def gen_3dbeam(maxh, nref, comm, lens = [10,1,1]):
    b = csg.OrthoBrick(csg.Pnt(-1,0,0), csg.Pnt(lens[0], lens[1], lens[2])).bc("other")
    p = csg.Plane(csg.Pnt(0,0,0), csg.Vec(-1,0,0)).bc("left")
    geo = csg.CSGeometry()
    geo.Add(b*p)
    return gen_ref_mesh(geo, maxh, nref, comm)


def gen_beam(dim, maxh, nref, comm, lens = None):
    if dim == 2:
        if lens is None:
            return gen_2dbeam(maxh, nref, comm)
        else:
            return gen_2dbeam(maxh, nref, comm, lens)
    else:
        if lens is None:
            return gen_3dbeam(maxh, nref, comm)
        else:
            return gen_3dbeam(maxh, nref, comm, lens)
        

def gen_sq_with_sqs(maxh, nref, comm):
    geo = ng.geom2d.SplineGeometry()

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

def setup_mcs (mesh, force = ngs.CoefficientFunction((0,0,0)), order = 1, sym = True, diriN = "", diriT = "", \
               el_int = False, divdiv=False, otrace=None, RT = False, a_opts = dict(), l2_coef = 0):
    hcd_opts = dict()
    if otrace is None:
        otrace = sym
    if RT:
        if sym and otrace:
            hcd_opts["ordertrace"] = order
        V = ngs.HDiv(mesh, RT=True, order=order, dirichlet=diriN, hodivfree=False)
        Vhat = ngs.TangentialFacetFESpace(mesh, order=order, dirichlet=diriT)
        Sigma = ngs.HCurlDiv(mesh, order=order, GGBubbles=True, discontinuous=True, **hcd_opts)
        # Sigma = HCurlDiv(mesh, order=order, orderinner=order+1, discontinuous=True, **hcd_opts) # slower
    else:
        if sym and otrace:
            hcd_opts["ordertrace"] = order - 1 # ?? seems to work...
        if sym and order < 2:
            raise "use order >= 2 for BDM + symmetric gradient version!"
        V = ngs.HDiv(mesh, RT=False, order=order, dirichlet=diriN, hodivfree=False)
        Vhat = ngs.TangentialFacetFESpace(mesh, order=order-1, dirichlet=diriT)
        Sigma = ngs.HCurlDiv(mesh, order=order-1, orderinner=order, discontinuous=True, **hcd_opts)
    Sigma.SetCouplingType(ngs.IntRange(0, Sigma.ndof), ngs.COUPLING_TYPE.HIDDEN_DOF)
    Sigma = ngs.Compress(Sigma)
    if sym:
        if mesh.dim == 2:
            S = ngs.L2(mesh, order=order - 1)
        else:
            S = ngs.VectorL2(mesh, order=order - 1)
        S.SetCouplingType(ngs.IntRange(0, S.ndof), ngs.COUPLING_TYPE.HIDDEN_DOF)
        S = ngs.Compress(S)
        X = ngs.FESpace([V, Vhat, Sigma, S])
        u, uhat, sigma, W = X.TrialFunction()
        v, vhat, tau, R = X.TestFunction()
    else:
        X = FESpace([V, Vhat, Sigma])
        u, uhat, sigma = X.TrialFunction()
        v, vhat, tau = X.TestFunction()
    dS = ngs.dx(element_boundary=True)
    n = ngs.specialcf.normal(mesh.dim)
    def tang(u):
        return u - (u * n) * n
    if mesh.dim == 2:
        def Skew2Vec(m):
            return m[1, 0] - m[0, 1]
    else:
        def Skew2Vec(m):
            return ngs.CoefficientFunction((m[0, 1] - m[1, 0], m[2, 0] - m[0, 2], m[1, 2] - m[2, 1]))
    nu = 1
    stokesA = -1/ nu * ngs.InnerProduct(sigma, tau) * ngs.dx + \
              (ngs.div(sigma) * v + ngs.div(tau) * u) * ngs.dx + \
              (-((sigma * n) * n) * (v * n) - ((tau * n) * n) * (u * n)) * dS + \
              (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS
    if divdiv:
        stokesA += nu * ngs.div(u) * ngs.div(v) * ngs.dx
    if sym:
        stokesA += (ngs.InnerProduct(W, Skew2Vec(tau)) + ngs.InnerProduct(R, Skew2Vec(sigma))) * ngs.dx
    if l2_coef != 0:
        stokesA += l2_coef * u * v * ngs.dx
    a = ngs.BilinearForm(X, eliminate_hidden = True, eliminate_inernal = True, **a_opts)
    a += stokesA
    f = ngs.LinearForm(X)
    f += ngs.InnerProduct(force, v) * ngs.dx
    return X, a, f

def setup_elast(mesh, mu = 1, lam = 0, nu = 1, E = None, f_vol = None, multidim = True,
                diri=".*", order=1, fes_opts=dict(), blf_opts=dict(), lf_opts=dict()):
    if E is not None:
        # convert Young's modulus E and poisson's ratio nu to lame parameters mu,lambda
        mu  = E / ( 2 * ( 1 + nu ))                        # eps-eps
        lam = ( E * nu ) / ( (1 + nu) * (1 - (2 * nu) ) )  # div-div

    dim = mesh.dim
    if multidim:
        V = ngs.H1(mesh, order=order, dirichlet=diri, **fes_opts, dim=mesh.dim)
    else:
        V = ngs.VectorH1(mesh, order=order, dirichlet=diri, **fes_opts)

    u,v = V.TnT()

    sym = lambda X : 0.5 * (X + X.trans)
    grd = lambda X : ngs.CoefficientFunction( tuple(ngs.grad(X)[i,j] for i in range(dim) for j in range(dim)), dims=(dim,dim))
    eps = lambda X : sym(grd(X))

    a = ngs.BilinearForm(V, symmetric=False, **blf_opts)
    a += mu * ngs.InnerProduct(eps(u), eps(v)) * ngs.dx

    if lam != 0:
        div = lambda U : sum([ngs.grad(U)[i,i] for i in range(1, dim)], start=ngs.grad(U)[0,0])
        a += lam * div(u) * div(v) * ngs.dx

    lf = ngs.LinearForm(V)
    lf += f_vol * v * ngs.dx

    return V, a, lf


def Solve(a, f, c, ms = 100, tol=1e-6, printIts=True, doTest=False, needsAssemble=True, printTimers=False, threading=False):
    gfu = ngs.GridFunction(a.space)

    comm = a.space.mesh.comm

    ts   = ngs.Timer("solve")
    tsup = ngs.Timer("setup")

    # turn off threading
    if not threading:
        ngs.ngsglobals.numthreads=1

    with ngs.TaskManager():

        if needsAssemble:

            ngs.ngsglobals.msg_level = 5
            comm.Barrier()
            tsup.Start()

            a.Assemble()

            ngs.ngsglobals.msg_level = 1
            comm.Barrier()
            tsup.Stop()

            f.Assemble()

        if doTest:
            c.Test()

        cb = None if a.space.mesh.comm.rank != 0 or not printIts else lambda k, x: print("it =", k , ", err =", x)
        cg = ngs.krylovspace.CGSolver(mat=a.mat, pre=c, callback = cb, maxiter=ms, tol=tol)

        comm.Barrier()
        ts.Start()
        cg.Solve(sol=gfu.vec, rhs=f.vec)
        comm.Barrier()
        ts.Stop()

        timeOf = lambda timer : timer.time * max(comm.size-1,1)

        if comm.rank == 0:
            print("---")
            print("multi-dim ", a.space.dim)
            print("(vectorial) ndof ", a.space.ndofglobal)
            print("(scalar) ndof ", a.space.dim * a.space.ndofglobal)
            print("used nits = ", cg.iterations)
            if needsAssemble:
                print(f"SETUP = {tsup.time} sec")
                print(f"(vec) dofs / (sec * np) = {a.space.ndofglobal / timeOf(tsup) }")
                print(f"(scal) dofs / (sec * np) = { (a.space.ndofglobal * a.space.dim) / timeOf(tsup) }")
            print(f"SOLVE = {ts.time} sec")
            print(f"(vec) dofs / (sec * np) =  {a.space.ndofglobal / timeOf(ts)}")
            print(f"(scal) dofs / (sec * np) = {(a.space.ndofglobal * a.space.dim) / timeOf(ts) }")
            print("---")

        if printTimers:
            minTime = 0.05 * (ts.time + tsup.time)
            sigTimers = []
            for timer in ngs.Timers():
                tt = timer["time"]
                if tt > minTime:
                    sigTimers.append( (timer["name"], tt) )
            sigTimers.sort(key = lambda x : -x[1])
            print("\n\n---")
            print("Significant time spent in:")
            for name, t in sigTimers:
                print(name, t)
            print("---\n\n")
            
        
    return gfu
