from ngsolve import *
import netgen, sys, ngs_amg
from netgen.meshing import MeshPoint, Pnt, FaceDescriptor, Element3D, Element2D

__VFEMB__ = None

def OneTet():
    if mpi_world.size > 1:
        raise 'simple mesh not parallel'
    ngmesh = netgen.meshing.Mesh()
    pids = [ ngmesh.Add(MeshPoint(Pnt(*t))) for t in [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]]
    ngmesh.Add(Element3D(1, pids))
    ngmesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    ngmesh.Add(Element2D(vertices=[1,2,4], index=1))
    ngmesh.Add(Element2D(vertices=[1,2,3], index=1))
    ngmesh.Add(Element2D(vertices=[1,3,4], index=1))
    ngmesh.Add(Element2D(vertices=[2,3,4], index=1))
    ngmesh.SetMaterial(1, "inner")
    mesh = Mesh(ngmesh)
    return mesh

def TwoTets():
    if mpi_world.size > 1:
        raise 'simple mesh not parallel'
    ngmesh = netgen.meshing.Mesh()
    rs = 2**(-1/2)
    pids = [ ngmesh.Add(MeshPoint(Pnt(*t))) for t in [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (rs,rs,rs)]]
    ngmesh.Add(Element3D(1, pids[:-1]))
    ngmesh.Add(Element3D(1, pids[1:]))
    ngmesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    ngmesh.Add(Element2D(vertices=[1,2,4], index=1))
    ngmesh.Add(Element2D(vertices=[1,2,3], index=1))
    ngmesh.Add(Element2D(vertices=[1,3,4], index=1))
    # ngmesh.Add(Element2D(vertices=[2,3,4], index=1))
    ngmesh.Add(Element2D(vertices=[5,2,4], index=1))
    ngmesh.Add(Element2D(vertices=[5,2,3], index=1))
    ngmesh.Add(Element2D(vertices=[5,3,4], index=1))
    ngmesh.SetMaterial(1, "inner")
    mesh = Mesh(ngmesh)
    return mesh


def SetUpTFSet(VF):
    mesh = VF.mesh
    global __VFEMB__
    if __VFEMB__ is not None:
        return
    vf_order = 0 if VF.ndof == 0 else VF.GetOrder(NodeId(ELEMENT, 0))
    L2 = VectorL2(mesh, vf_order)
    u,v = VF.TnT()
    M = BilinearForm(VF)
    M += InnerProduct(u,v) * dx(element_boundary=True)
    M.Assemble()
    uh,vh = L2.TnT()
    M2 = BilinearForm(trialspace=L2, testspace=VF)
    M2 += InnerProduct(uh,v) * dx(element_boundary=True)
    M2.Assemble()
    Minv = M.mat.Inverse()
    __VFEMB__ = [L2, GridFunction(L2), Minv @ M2.mat]


def SetVF(gfh, cf):
    SetUpTFSet(gfh.space)
    L2, gfl2, T = __VFEMB__
    gfl2.Set(cf)
    gfh.vec.data = T * gfl2.vec
    
def TestRBMS(V, A, pc, draw = False):
    if draw:
        mesh = V.mesh
        ms = Draw(mesh, name="mesh")
    rb_gfs = list()
    rb_cfs = [(1,0,0), (0,1,0), (0,0,1), (0,-z,y), (z,0,-x), (-y,x,0)]
    for k,cf in enumerate(rb_cfs):
        g = GridFunction(V)
        g.components[0].Set(cf)
        SetVF(g.components[1], cf)
        # g.components[1].Set(cf)
        rb_gfs.append(g)
    vw_rbms = pc.GetRBModes()
    A_aux = pc.aux_mat
    P = pc.P
    t_V = A.CreateColVector()
    t_V2 = A.CreateColVector()
    t_V3 = A.CreateColVector()
    t_A = pc.CreateAuxVector()
    for k,(v,w) in enumerate(zip(*vw_rbms)):
        sys.stdout.flush()
        print('- check RBM nr.', k)

        print('-- check energies: ')
        t_A.data = A_aux * w
        sqe_w = InnerProduct(t_A,w)
        t_V.data = A * v
        sqe_v = InnerProduct(t_V, v)
        print('---- AUX energy sq', sqe_w)
        print('---- AUX energy   ', sqe_w**0.5)
        print('----  V  energy sq', sqe_v)
        print('----  V  energy   ', sqe_v**0.5)

        print('-- check diff ( i think broken): ')
        t_V2 = t_V
        t_V -= rb_gfs[k].vec
        print('--- || rb - v ||', Norm(t_V))
        t_V2.data = rb_gfs[k].vec
        t_V2 -= P * w
        print('--- || rb - Pw ||', Norm(t_V2))
        t_V3.data = v - P * w
        print('--- || v - Pw ||', Norm(t_V3))

        print('\n')

        if draw:
            gfv = GridFunction(V)
            gfv.vec.data = v
            gfw = GridFunction(V)
            gfw.vec.data = P * w
            if ms is not None: # new gui
                scs = list()
                def add_scene(g, name, d, dd):
                    if d:
                        s = Draw(g, name=name)
                        s.active = False
                        scs.append(s)
                    if dd:
                        s2 = Draw(mesh, deformation=g, name="DEFO_"+name)
                        s2.setDeformation(True)
                        s2.setDeformationScale(1.0)
                        s2.active = False
                        scs.append(s2)
                add_scene(gfv.components[0], "HD_V_rb"+str(k), True, True)
                add_scene(gfv.components[1], "VF_V_rb"+str(k), False, True)
                # add_scene(gfw.components[0], "HD_W_rb"+str(k), True, True)
                # add_scene(gfw.components[1], "VF_W_rb"+str(k), False, True)
                add_scene(CoefficientFunction(gfv.components[0] + gfv.components[1]), "S_rb"+str(k), False, True)
            else: # netgen gui
                Draw(gfv.components[0], name="HD_V_rb"+str(k))
                # Draw(gfw.components[1], name="VF_W_rb"+str(k))
                Draw(gfv.components[0], name="HD_V_rb"+str(k))
                # Draw(gfw.components[1], name="VF_W_rb"+str(k))
