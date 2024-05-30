import ngsolve as ngs
import netgen as ng
import netgen.geom2d as geom2d
import netgen.csg as csg
import ngsolve.krylovspace

def MakeStructured3DMesh(hexes=False, nx=10, ny=None, nz=None, secondorder=False,
                         periodic_x=False, periodic_y=False, periodic_z=False, mapping = None, cuboid_mapping=False,
                         get_index = lambda x,y,z : 1, mat_names = ["default"]):
    if nz == None:
        if ny == None:
            nz = nx
        else:
            raise("MakeStructured3DMesh: No default value for nz if nx and ny are provided")
    if ny == None:
        ny = nx
    netmesh = NGMesh()
    netmesh.dim = 3
    if cuboid_mapping:
        P1 = (0,0,0)
        P2 = (1,1,1)
        if mapping:
            P1 = mapping(*P1)
            P2 = mapping(*P2)
        cube = OrthoBrick(Pnt(P1[0], P1[1], P1[2]), Pnt(P2[0], P2[1], P2[2])).bc(1)
        geom = CSGeometry()
        geom.Add(cube)
        netmesh.SetGeometry(geom)
    pids = []
    if periodic_x:
        slavei = []
        masteri = []
    if periodic_y:        
        slavej = []
        masterj = []
    if periodic_z:        
        slavek = []
        masterk = []        
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                # x,y,z = mapping(i / nx, j / ny, k / nz)
                x,y,z = i / nx, j / ny, k / nz
                # if mapping:
                #   x,y,z = mapping(x,y,z)
                pids.append(netmesh.Add(MeshPoint(Pnt( x,y,z ))))
                if periodic_x:
                    if i == 0:
                        slavei.append(pids[-1])
                    if i == nx:
                        masteri.append(pids[-1])  
                if periodic_y:           
                    if j == 0:
                        slavej.append(pids[-1])
                    if j == ny:
                        masterj.append(pids[-1]) 
                if periodic_z:                    
                    if k == 0:
                        slavek.append(pids[-1])
                    if k == nz:
                        masterk.append(pids[-1])
    if periodic_x:
        for i in range(len(slavei)):   
            netmesh.AddPointIdentification(masteri[i],slavei[i],identnr=1,type=2)     
    if periodic_y:        
        for j in range(len(slavej)):            
            netmesh.AddPointIdentification(masterj[j],slavej[j],identnr=2,type=2) 
    if periodic_z:        
        for k in range(len(slavek)):            
            netmesh.AddPointIdentification(masterk[k],slavek[k],identnr=3,type=2)                                                      

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                base = i * (ny+1)*(nz+1) + j*(nz+1) + k
                baseup = base+(ny+1)*(nz+1)
                eps = 1e-5
                pnum = [base, base+1, base+(nz+1)+1, base+(nz+1),
                        baseup, baseup+1, baseup+(nz+1)+1, baseup+(nz+1)]
                index = get_index((i+0.5) / nx, (j+0.5) / ny, (k+0.5) / nz)
                if hexes:
                    elpids = [pids[p] for p in pnum]
                    el = Element3D(index, elpids)
                    if not mapping:
                        el.curved = False
                    netmesh.Add(el)
                else:
                    #  a poor mans kuhn triangulation of a cube
                    for qarr in [[0, 4, 5, 6],
                                 [0, 6, 7, 4],
                                 [0, 3, 7, 6],
                                 [0, 1, 6, 5],
                                 [0, 1, 2, 6],
                                 [0, 3, 6, 2]]:
                        elpids = [pids[p] for p in [pnum[q] for q in qarr]]
                        
                        netmesh.Add(Element3D(index, elpids))

    def AddSurfEls(p1, dxi, nxi, deta, neta, facenr):
        def add_seg(i, j, os):
            base = p1 + i*dxi + j*deta
            pnum = [base, base+os]
            elpids = [pids[p] for p in pnum]
            netmesh.Add(Element1D(elpids, index=facenr))
        for i in range(nxi):
            for j in [0,neta]:
                add_seg(i,j,dxi)
        for i in [0,nxi]:
            for j in range(neta):
                add_seg(i,j,deta)
        for i in range(nxi):
            for j in range(neta):
                base = p1 + i*dxi+j*deta
                pnum = [base, base+dxi, base+dxi+deta, base+deta]
                if hexes:
                    elpids = [pids[p] for p in pnum]
                    netmesh.Add(Element2D(facenr, elpids))
                else:
                    qarrs = [[0, 1, 2], [0, 2, 3]]
                    for qarr in qarrs:
                        elpids = [pids[p] for p in [pnum[q] for q in qarr]]
                        netmesh.Add(Element2D(facenr, elpids))

    #order is important!
    netmesh.Add(FaceDescriptor(surfnr=4, domin=1, bc=1))
    netmesh.Add(FaceDescriptor(surfnr=2, domin=1, bc=2))
    netmesh.Add(FaceDescriptor(surfnr=5, domin=1, bc=3))
    netmesh.Add(FaceDescriptor(surfnr=3, domin=1, bc=4))
    netmesh.Add(FaceDescriptor(surfnr=0, domin=1, bc=5))
    netmesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=6))
        
    # y-z-plane, smallest x-coord: ("back")
    AddSurfEls(0, 1, nz,  nz+1, ny, facenr=1) # y-z-plane
    # x-z-plane, smallest y-coord: ("left")
    AddSurfEls(0, (ny+1)*(nz+1), nx, 1, nz,facenr=2)
    # y-z-plane, largest x-coord: ("front")
    AddSurfEls((nx+1)*(ny+1)*(nz+1)-1, -(nz+1), ny, -1, nz, facenr=3) 
    # x-z-plane, largest y-coord: ("right")
    AddSurfEls((nx+1)*(ny+1)*(nz+1)-1, -1, nz, -(ny+1)*(nz+1), nx, facenr=4)
    # x-y-plane, smallest z-coord: ("bottom")
    AddSurfEls(0, nz+1, ny, (ny+1)*(nz+1), nx,facenr=5) 
    # x-y-plane, largest z-coord: ("top")
    AddSurfEls((nx+1)*(ny+1)*(nz+1)-1, -(ny+1)*(nz+1), nx, -(nz+1), ny, facenr=6) 

    for k,name in enumerate(mat_names):
        netmesh.SetMaterial(k+1, name)
    
    if cuboid_mapping:
        netmesh.SetBCName(0,"back")
        netmesh.SetBCName(1,"left")
        netmesh.SetBCName(2,"front")
        netmesh.SetBCName(3,"right")
        netmesh.SetBCName(4,"bottom")
        netmesh.SetBCName(5,"top")
    
    netmesh.Compress()

    if secondorder:
        netmesh.SecondOrder()
    
    if mapping:
        for p in netmesh.Points():
            x,y,z = p.p
            x,y,z = mapping(x,y,z)
            p[0] = x
            p[1] = y
            p[2] = z

    return netmesh



def MakeStructured2DMesh(quads=True, nx=10, ny=10, secondorder=False, periodic_x=False, periodic_y=False, mapping = None,
                         get_index = lambda x,y : 1, mat_names = ["default"]):
    mesh = NGMesh()
    mesh.dim = 2

    pids = []
    if periodic_y:
        slavei = []
        masteri = []
    if periodic_x:        
        slavej = []
        masterj = []
    for i in range(ny+1):
        for j in range(nx+1):
            x,y = j/nx, i/ny
            # if mapping:
            #    x,y = mapping(x,y)
            pids.append(mesh.Add (MeshPoint(Pnt(x,y,0))))
            if periodic_y:
                if i == 0:
                    slavei.append(pids[-1])
                if i == ny:
                    masteri.append(pids[-1])  
            if periodic_x:                       
                if j == 0:
                    slavej.append(pids[-1])
                if j == nx:
                    masterj.append(pids[-1])        
    if periodic_y:
        for i in range(len(slavei)):   
            mesh.AddPointIdentification(masteri[i],slavei[i],identnr=1,type=2)
    if periodic_x:            
        for j in range(len(slavej)):        
            mesh.AddPointIdentification(masterj[j],slavej[j],identnr=2,type=2)                                       

    mesh.Add(FaceDescriptor(surfnr=1,domin=1,domout=0, bc=1))
    mesh.Add(FaceDescriptor(surfnr=2,domin=2,bc=1, domout=1))
    mesh.SetMaterial(1, "mat")
    mesh.SetMaterial(2, "fiber")

    for i in range(ny):
        for j in range(nx):
            base = i * (nx+1) + j
            x, y = (j+0.5) / nx, (i+0.5) / ny
            if mapping:
                x,y = mapping(x,y)
            index = get_index(x,y)
            print('ind', index)
            if quads:
                pnum = [base,base+1,base+nx+2,base+nx+1]
                elpids = [pids[p] for p in pnum]
                el = Element2D(vertices=elpids,index=index)
                if not mapping:
                    el.curved=False
                mesh.Add(el)
            else:
                pnum1 = [base,base+1,base+nx+1]
                pnum2 = [base+1,base+nx+2,base+nx+1]
                elpids1 = [pids[p] for p in pnum1]
                elpids2 = [pids[p] for p in pnum2]
                mesh.Add(Element2D(vertices=elpids1,index=index)) 
                mesh.Add(Element2D(vertices=elpids2,index=index))                          

    
    for i in range(nx):
        mesh.Add(Element1D([pids[i], pids[i+1]], index=1))
    for i in range(ny):
        mesh.Add(Element1D([pids[i*(nx+1)+nx], pids[(i+1)*(nx+1)+nx]], index=2))
    for i in range(nx):
        mesh.Add(Element1D([pids[ny*(nx+1)+i+1], pids[ny*(nx+1)+i]], index=3))
    for i in range(ny):
        mesh.Add(Element1D([pids[(i+1)*(nx+1)], pids[i*(nx+1)]], index=4))

    mesh.SetBCName(0, "bottom")        
    mesh.SetBCName(1, "right")        
    mesh.SetBCName(2, "top")        
    mesh.SetBCName(3, "left")  

    mesh.Compress()       
    
    if secondorder:
        mesh.SecondOrder()
    
    if mapping:
        for p in mesh.Points():
            x,y,z = p.p
            x,y = mapping(x,y)
            p[0] = x
            p[1] = y
            
    return mesh

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

def setup_norot_elast(mesh, mu = 1, lam = 0, f_vol = None, multidim = True, reorder = False,
                      diri=".*", order=1, fes_opts=dict(), blf_opts=dict(), lf_opts=dict()):
    dim = mesh.dim
    if multidim:
        V = ngs.H1(mesh, order=order, dirichlet=diri, **fes_opts, dim=mesh.dim)
    else:
        V = ngs.VectorH1(mesh, order=order, dirichlet=diri, **fes_opts)

    if reorder:
        V = ngs.comp.Reorder(V)
        
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


def setup_rot_elast(mesh, mu = 1, lam = 0, f_vol = None, multidim = True, reorder = False,
                    diri=".*", order=1, fes_opts=dict(), blf_opts=dict(), lf_opts=dict()):
    dim = mesh.dim
    mysum = lambda x : sum(x[1:], x[0])
    if dim == 2:
        to_skew = lambda x : ngs.CoefficientFunction( (0, -x[0], x[0], 0), dims = (2,2) )
    else:
        to_skew = lambda x : ngs.CoefficientFunction( (  0   , x[2],  -x[1], \
                                                         -x[2],    0 , x[0], \
                                                         x[1], -x[0],   0), dims = (3,3) )
    if multidim:
        mdim = dim + ( (dim-1) * dim) // 2
        V = ngs.H1(mesh, order=order, dirichlet=diri, **fes_opts, dim=mdim)
        if reorder:
            V = ngs.comp.Reorder(V)
        trial, test = V.TnT()
        u = ngs.CoefficientFunction( tuple(trial[x] for x in range(dim)) )
        gradu = ngs.CoefficientFunction( tuple(ngs.Grad(trial)[i,j] for i in range(dim) for j in range(dim)), dims = (dim, dim))
        divu = mysum( [ngs.Grad(trial)[i,i] for i in range(dim)] )
        w = to_skew([trial[x] for x in range(dim, mdim)])
        ut = ngs.CoefficientFunction( tuple(test[x] for x in range(dim)) )
        gradut = ngs.CoefficientFunction( tuple(ngs.Grad(test)[i,j] for i in range(dim) for j in range(dim)), dims = (dim, dim))
        divut = mysum( [ngs.Grad(test)[i,i] for i in range(dim)] )
        wt = to_skew([test[x] for x in range(dim, mdim)])
    else:
        Vu = ngs.VectorH1(mesh, order=order, dirichlet=diri, **fes_opts)
        if reorder == "sep":
            Vu = ngs.comp.Reorder(Vu)
        if dim == 3:
            Vw = Vu
        else:
            Vw = ngs.H1(mesh, order=order, dirichlet=diri, **fes_opts)
            if reorder == "sep":
                Vw = ngs.comp.Reorder(Vw)
        V = ngs.FESpace([Vu, Vw])
        # print("free pre RO: ", V.FreeDofs())
        if reorder is True:
            V = ngs.comp.Reorder(V)
        # print("free post RO: ", V.FreeDofs())
        (u,w), (ut, wt) = V.TnT()
        gradu = ngs.Grad(u)
        divu = mysum( [ngs.Grad(u)[i,i] for i in range(dim)] )
        w = to_skew(w)
        gradut = ngs.Grad(ut)
        divut = mysum( [ngs.Grad(ut)[i,i] for i in range(dim)] )
        wt = to_skew(wt)

    a = ngs.BilinearForm(V, **blf_opts)
    a += ( mu * ngs.InnerProduct(gradu - w, gradut - wt) ) * ngs.dx
    #a += ngs.InnerProduct(w,wt) * ngs.dx

    #trial, test = V.TnT()
    #a += 0.1 * ngs.InnerProduct(trial,test) * ngs.dx
    
    if lam != 0:
        a += lam * divu * divut * ngs.dx

    lf = ngs.LinearForm(V)
    lf += f_vol * ut * ngs.dx

    return V, a, lf


def setup_elast(mesh, mu = 1, lam = 0, f_vol = None, multidim = True, rotations = False, reorder=False,
                diri=".*", order=1, fes_opts=dict(), blf_opts=dict(), lf_opts=dict()):
    if rotations:
        return setup_rot_elast(mesh, mu, lam, f_vol, multidim, reorder, diri, order, fes_opts, blf_opts, lf_opts)
    else:
        return setup_norot_elast(mesh, mu, lam, f_vol, multidim, reorder, diri, order, fes_opts, blf_opts, lf_opts)



def Solve(a, f, c, ms = 100, tol=1e-6, nocb=True):
    gfu = ngs.GridFunction(a.space)
    with ngs.TaskManager():
        ngs.ngsglobals.msg_level = 5
        a.Assemble()
        ngs.ngsglobals.msg_level = 1
        f.Assemble()
        c.Test()
        cb = None if a.space.mesh.comm.rank != 0 or nocb else lambda k, x: print("it =", k , ", err =", x)
        cg = ngs.krylovspace.CGSolver(mat=a.mat, pre=c, callback = cb, maxsteps=ms, tol=tol)
        ngs.NG_MPI_world.Barrier()
        ts = ngs.NG_MPI_world.WTime()
        cg.Solve(sol=gfu.vec, rhs=f.vec)
        ngs.NG_MPI_world.Barrier()
        ts = ngs.NG_MPI_world.WTime() - ts
        if ngs.NG_MPI_world.rank == 0:
            print("---")
            print("multi-dim ", a.space.dim)
            print("(vectorial) ndof ", a.space.ndofglobal)
            print("(scalar) ndof ", a.space.dim * a.space.ndofglobal)
            print("used nits = ", cg.iterations)
            print("(vec) dofs / (sec * np) = ", a.space.ndofglobal / (ts * max(ngs.NG_MPI_world.size-1,1)))
            print("(scal) dofs / (sec * np) = ", (a.space.ndofglobal * a.space.dim) / (ts * max(ngs.NG_MPI_world.size-1,1)))
            print("---")
        
    assert cg.errors[-1] < tol * cg.errors[0]
    assert cg.iterations < ms
    return gfu
