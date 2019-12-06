from ngsolve import *
from amg_utils import *
import netgen,sys,ngs_amg

SetTestoutFile('test.out')

def MatIter (amat, n_vecs = 5, lam_max = 1, lam_min = 0, reverse = True, M = 1e5, startvec = None, tol=1e-8):
    A = IdentityMatrix(amat.height) - 1/(lam_max-lam_min) * amat if reverse else amat
    evecs = list()
    evals = list()
    if startvec is None:
        startvec = amat.CreateColVector()
        startvec.Cumulate()
        import random
        for k in range(len(startvec)):
            startvec[k] = random.randint(1,1000) / 1000
        startvec.Distribute()
        startvec.Cumulate()
    startvec *= InnerProduct(startvec, startvec)**(-0.5)
    # print('inin', Norm(startvec))
    tempvec = startvec.CreateVector()
    errvec = startvec.CreateVector()
    def ortho(vec, base):
        for k,bv in enumerate(base):
            ip = InnerProduct(vec, bv)
            vec -= ip * bv
    for K in range(n_vecs):
        for l in range(int(M)):
            tempvec.data = A * startvec
            ortho(tempvec, evecs)
            ip = InnerProduct(tempvec, startvec)
            errvec.data = tempvec - ip * startvec
            # print('norms', Norm(startvec), Norm(tempvec), Norm(errvec))
            startvec.data = 1/Norm(tempvec) * tempvec
            err = Norm(errvec)
            if err < tol * ip:
                break
        if reverse:
            print('evec', K, 'after', l, 'its with err', err, 'and eval', ip, 'orig eval', (lam_max - lam_min) * ( 1 - ip))
        else:
            print('evec', K, 'after', l, 'its with err', err, 'and eval', ip)
        vk = startvec.CreateVector()
        vk.data = startvec
        evecs.append(vk)
        evals.append(ip)
    return evecs, evals
    
        
def SetUpMCS (mesh, force = CoefficientFunction((0,0,0)), order = 1, sym = True, diriN = "", diriT = "", el_int = False, divdiv=False, otrace=None,
              RT = False):
    hcd_opts = dict()
    if otrace is None:
        otrace = sym
    if sym and otrace:
        hcd_opts["ordertrace"] = order - 1
    if RT:
        V = HDiv(mesh, RT=True, order=order, dirichlet=diriN, hodivfree=False)
        Vhat = TangentialFacetFESpace(mesh, order=order, dirichlet=diriT)
        Sigma = HCurlDiv(mesh, order=order, GGBubbles=True, discontinuous=True, **hcd_opts)
        # Sigma = HCurlDiv(mesh, order=order, orderinner=order+1, discontinuous=True, **hcd_opts) # slower
    else:
        if sym and order < 2:
            raise "use order >= 2 for BDM + symmetric gradient version!"
        V = HDiv(mesh, RT=False, order=order, dirichlet=diriN, hodivfree=False)
        Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet=diriT)
        Sigma = HCurlDiv(mesh, order=order-1, orderinner=order, discontinuous=True, **hcd_opts)
    Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
    Sigma = Compress(Sigma)
    if sym:
        if mesh.dim == 2:
            S = L2(mesh, order=order - 1)
        else:
            S = VectorL2(mesh, order=order - 1)
        S.SetCouplingType(IntRange(0, S.ndof), COUPLING_TYPE.HIDDEN_DOF)
        S = Compress(S)
        X = FESpace([V, Vhat, Sigma, S])
        u, uhat, sigma, W = X.TrialFunction()
        v, vhat, tau, R = X.TestFunction()
    else:
        X = FESpace([V, Vhat, Sigma])
        u, uhat, sigma = X.TrialFunction()
        v, vhat, tau = X.TestFunction()
    dS = dx(element_boundary=True)
    n = specialcf.normal(mesh.dim)
    def tang(u):
        return u - (u * n) * n
    if mesh.dim == 2:
        def Skew2Vec(m):
            return m[1, 0] - m[0, 1]
    else:
        def Skew2Vec(m):
            return CoefficientFunction((m[0, 1] - m[1, 0], m[2, 0] - m[0, 2], m[1, 2] - m[2, 1]))
    nu = 1
    stokesA = -1/ nu * InnerProduct(sigma, tau) * dx + \
              (div(sigma) * v + div(tau) * u) * dx + \
              (-((sigma * n) * n) * (v * n) - ((tau * n) * n) * (u * n)) * dS + \
              (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS
    if divdiv:
        stokesA += nu * div(u) * div(v) * dx
    if sym:
        stokesA += (InnerProduct(W, Skew2Vec(tau)) + InnerProduct(R, Skew2Vec(sigma))) * dx
    a = BilinearForm(X, eliminate_hidden = True, eliminate_inernal = True, elmatev = True)
    a += stokesA
    f = LinearForm(X)
    f += InnerProduct(force, v) * dx
    return X, a, f


order = 1
geo, maxh = netgen.csg.unit_cube, 0.4
el_int = True
geo, mesh = gen_ref_mesh (geo, maxh, nref = 0, comm=mpi_world, mesh_file = '', save = False)
sym = True
diri = "back"
dovi = False
doso = False

print(mesh.GetBoundaries())

X, a, f = SetUpMCS(mesh, order = order, force = CoefficientFunction((0,-x*(1-x),0)), diriN = diri, diriT = diri, el_int = el_int, sym = sym,
                   divdiv=False, otrace=True, RT=True)

#with TaskManager():
a.Assemble()
f.Assemble()

ainv = a.mat.Inverse(X.FreeDofs(el_int), inverse="umfpack")
gfu = GridFunction(X)
gfu.vec.data = ainv * f.vec

def defo(gf):
    return CoefficientFunction(gf.components[0] + gf.components[1])

s = Draw(mesh, deformation=defo(gfu), name='sol')
if s is not None:
    s.setDeformation(True)
    s.setDeformationScale(1)
else:
    Draw(gfu.components[0], mesh, 'u')
    Draw(gfu.components[1], mesh, 'uhat')


n = specialcf.normal(mesh.dim)
def tang(u):
    return u - (u * n) * n
def normal(u):
    return (u * n) * n


Vvis = VectorH1(mesh, order=order)
uh1,vh1 = Vvis.TnT()
M = BilinearForm(Vvis)
M += InnerProduct(uh1,vh1) * dx
M.Assemble()
if sym:
    u, uhat, sigma, W = X.TrialFunction()
    v, vhat, tau, R = X.TestFunction()
else:
    u, uhat, sigma = X.TrialFunction()
    v, vhat, tau = X.TestFunction()
MI = M.mat.Inverse()
MM = BilinearForm(trialspace=X, testspace=Vvis)
MM += InnerProduct(u, vh1) * dx
MM += InnerProduct(uhat, vh1) * dx(element_boundary=True)
MM.Assemble()
M2 = BilinearForm(X)
M2 += InnerProduct(u,v) * dx
M2 += InnerProduct(uhat,tang(vhat)) * dx(element_boundary=True)
M2.Assemble()
MI2 = M2.mat.Inverse()
MM2 = BilinearForm(trialspace=Vvis, testspace=X)
MM2 += InnerProduct(v, uh1) * dx
MM2 += InnerProduct(vhat, tang(uh1)) * dx(element_boundary=True)
MM2.Assemble()
T  = MI @ MM.mat
T2 = MI2 @ MM2.mat
#print(MM.mat)
#quit()
gfv = GridFunction(Vvis)
# print('gfu', InnerProduct(gfu.vec, gfu.vec))
# gfv.vec.data = MM.mat * gfu.vec
# print('gfv1', InnerProduct(gfv.vec, gfv.vec))
# gfv.vec.data = T * gfu.vec
# print('gfv2', InnerProduct(gfv.vec, gfv.vec))
# Draw(gfv, mesh, 'VIS')

gf_x = GridFunction(X)
defo = CoefficientFunction((-y,x,1))

print('MM2', MM2.mat.height, MM2.mat.width)
print('gf_x', len(gf_x.vec))
print('gfv', len(gfv.vec))

gfv.Set(defo)
gf_x.vec.data = T2 * gfv.vec
# gf_x.components[0].Set(defo)


# print(X.components)
# OS = X.components[0].ndof
# for f in mesh.facets:
#     dnums = X.components[1].GetDofNrs(f)
#     gf_x.vec[OS + dnums[0]] = 1
#     gf_x.vec[OS + dnums[1]] = 1

#print(gf_x.vec)
#quit()
t = gf_x.vec.CreateVector()
t.data = a.mat * gf_x.vec
print(len(t), InnerProduct(t,t))


if doso:
    ai = a.mat.Inverse(X.FreeDofs(el_int))
    gfsol = GridFunction(X)
    gfsol.vec.data = ai * f.vec
    Draw(gfsol.components[0], mesh, 'HD_SOL')
    Draw(gfsol.components[1], mesh, 'VF_SOL')
    

if dovi:
    startvec = a.mat.CreateColVector()
    startvec.Cumulate()
    import random
    for k in range(len(startvec)):
        startvec[k] = random.randint(1,1000) / 1000
    startvec.Distribute()
    startvec.Cumulate()
    fP = Projector(X.FreeDofs(el_int), True)
    fP.Project(startvec)
    amf = a.mat if diri == "" else fP @ a.mat

    with TaskManager():
        evecs, evals = MatIter(amf, n_vecs = 1, reverse=True, lam_max = 112, tol=1e-10, M=1e6, startvec=startvec)

    print('evals:', evals)

    for k,v in enumerate(evecs):
        gf = GridFunction(X)
        gf.vec.data = v
        Draw(gf.components[0], mesh, 'HD_BAD'+str(k))
        Draw(gf.components[1], mesh, 'VF_BAD'+str(k))
        tv = gf.vec.CreateVector()
        tv.data = a.mat * gf.vec
        tv2 = gf.vec.CreateVector()
        tv2.data = fP * tv
        print('norms', Norm(tv), Norm(tv2))
    
