from ngsolve import *
import netgen as ng
import netgen.geom2d as geom2d
import netgen.csg as csg

def geo_2dchannel(H=0.41, L=2.2, obstacle=True):
    geo = geom2d.SplineGeometry()
    geo.AddRectangle((0, 0), (L,H), bcs=("wall", "outlet", "wall", "inlet"))
    uin = CoefficientFunction( (1.5 * (2/H)**2 * y * (H - y), 0))
    if obstacle == True:
        pos = 0.2
        r = 0.05
        geo.AddCircle((pos, pos), r=r, leftdomain=0, rightdomain=1, bc="wall")
    return uin, geo

def geo_3dchannel(H=0.41, L=0.41, W=2.2, obstacle=True):
    geo = csg.CSGeometry()
    channel = csg.OrthoBrick( csg.Pnt(-1, 0, 0), csg.Pnt(L+1, H, W) ).bc("wall")
    inlet = csg.Plane (csg.Pnt(0,0,0), csg.Vec(-1,0,0)).bc("inlet")
    outlet = csg.Plane (csg.Pnt(L, 0,0), csg.Vec(1,0,0)).bc("outlet")
    uin = CoefficientFunction( (2.25 * (2/H)**2 * (2/W)**2 * y * (H - y) * z * (W - z), 0, 0))
    if obstacle == True:
        pos = (0.5, 0.2)
        r = 0.05
        cyl = csg.Cylinder(csg.Pnt(pos[0], pos[1], 0), csg.Pnt(pos[0], pos[1], 1), r).bc("wall")
        fluiddom = channel*inlet*outlet-cyl
    else:
        fluiddom = channel*inlet*outlet
    geo.Add(fluiddom)
    return uin, geo



def StokesHDGDiscretizationCompound(mesh, order, inlet="", wall="", outlet="", hodivfree=True, proj_jumps=True, div_div_pen=None, with_pressure=True, V=None, Vhat=None, nu=1, diri=f"wall|inlet"):
    if V is None:
        V1 = HDiv ( mesh, order = order, dirichlet = diri, hodivfree=hodivfree )
        V2 = TangentialFacetFESpace(mesh, order = order, dirichlet = diri, highest_order_dc=proj_jumps )
    else:
        V1 = V
        V2 = Vhat

    if with_pressure:
        Q = L2( mesh, order = 0 if hodivfree else order-1)
        W = V1 * V2 * Q
        u, uhat, p = W.TrialFunction()
        v, vhat, q = W.TestFunction()
    else:
        W = V1 * V2
        u, uhat = W.TrialFunction()
        v, vhat = W.TestFunction()

    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    def tang(vec):
        return vec - (vec*n)*n

    alpha = 3  # SIP parameter
    dS = dx(element_boundary=True)
    a = BilinearForm ( W)
    a += nu * InnerProduct ( Grad(u), Grad(v) ) * dx
    a += nu * InnerProduct ( Grad(u)*n, tang(vhat-v) ) * dS
    a += nu * InnerProduct ( Grad(v)*n, tang(uhat-u) ) * dS
    a += nu * alpha*order*order/h * InnerProduct(tang(vhat-v), tang(uhat-u)) * dS
    if div_div_pen is not None:
        a += div_div_pen * nu * InnerProduct(div(u), div(v)) * dx
    if with_pressure:
        a += (-div(u)*q - div(v)*p) * dx

    return W, a


def StokesHDGDiscretization(mesh, order, inlet="", wall="", outlet="", hodivfree=True, proj_jumps=True, div_div_pen=None, with_pressure=True, V=None, Vhat=None, nu=1, diri=f"wall|inlet"):
    V1 = HDiv ( mesh, order = order, dirichlet = diri, hodivfree=hodivfree )
    V2 = TangentialFacetFESpace(mesh, order = order, dirichlet = diri, highest_order_dc=proj_jumps )
    V = V1 * V2

    Q = L2( mesh, order = 0 if hodivfree else order-1)

    (u, uhat), (v, vhat) = V.TnT()
    p,q = Q.TnT()

    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    def tang(vec):
        return vec - (vec*n)*n

    alpha = 3  # SIP parameter
    dS = dx(element_boundary=True)

    a = BilinearForm ( V )
    a += nu * InnerProduct ( Grad(u), Grad(v) ) * dx
    a += nu * InnerProduct ( Grad(u)*n, tang(vhat-v) ) * dS
    a += nu * InnerProduct ( Grad(v)*n, tang(uhat-u) ) * dS
    a += nu * alpha*order*order/h * InnerProduct(tang(vhat-v), tang(uhat-u)) * dS

    if div_div_pen is not None:
        aPen = BilinearForm ( V )
        aPen += nu * InnerProduct ( Grad(u), Grad(v) ) * dx
        aPen += nu * InnerProduct ( Grad(u)*n, tang(vhat-v) ) * dS
        aPen += nu * InnerProduct ( Grad(v)*n, tang(uhat-u) ) * dS
        aPen += nu * alpha*order*order/h * InnerProduct(tang(vhat-v), tang(uhat-u)) * dS
        aPen += div_div_pen * nu * InnerProduct(div(u), div(v)) * dx
    else:
        aPen = a

    b = BilinearForm(trialspace=V, testspace=Q)
    b += -div(u)*q * dx

    return V, Q, a, b, aPen