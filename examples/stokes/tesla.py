from ngsolve import *
from netgen.occ import *

def GetValve(N, dim, R = 1.4, alpha = 45, Winlet = 1, beta = 190, L1 = 11, L2 = 6, Linlet = 5, closevalve = False):
    alphar = alpha * pi/180
    W = 1/cos(alphar/2)
    facW = 1
    Winlet2 =  Winlet
        
    wp = WorkPlane()
    p1 = 0
    p2 = 0

    wp.MoveTo(p1,p1) 
    r1 = Rectangle(L1,W).Face()
    
    r2 = wp.MoveTo(p1,p2+W).Rotate(-90+alpha).Move(W).Rotate(90).Rectangle(L2,W).Face()
    
    wp.MoveTo(p1,p2+W).Move(L2)
    c1 = Face(wp.Arc(W + R, -beta).Line(L1).Rotate(-90).Line(W).Rotate(-90).Line(L1).Arc(R,beta).Close().Wire())
    wp.MoveTo(0,W).Direction(1,0)
    cutplane = Rectangle(2*L1,4*L1).Face()
    

    v1 = r1 + r2  + (cutplane*c1) 
    
    ll = L1 + L1 * cos(alphar) - cos(alphar) * W
    hh = L1 * sin(alphar) - (1-sin(alphar)) * W
    didi = sqrt((L1 + L1 * cos(alphar))**2 + (L1 * sin(alphar))**2 ) - 2*W * sin(alphar/2) 
    
    dd = gp_Dir(cos(alpha), sin(alpha),0)
    v2 = v1.Mirror(Axis((0,0,0), X)).Move((0,W,0)).Rotate(Axis((0,0,0), Z), alpha).Move((L1,0,0))

    onevalve = (v1 + v2.Reversed()).Move((0,-W/2,0)).Rotate(Axis((0,0,0), Z), -alpha/2)

    vv = onevalve
    
    for i in range(1,N):
        vv = onevalve.Move((didi * i ,0, 0)) + vv
        
    
    
    inlet = wp.MoveTo(-Linlet,-Winlet2/2).Rectangle(Linlet* facW + W * sin(alphar/2) ,Winlet2).Face()
    outlet = wp.MoveTo(didi*N + W/2 * sin(alphar/2)* facW,-Winlet2/2).Rectangle(Linlet ,Winlet2).Face()
    vv = inlet + vv + outlet
    
    
    if closevalve == True:
        c1_x = -Linlet
        c1_y = -Winlet2/2

        c2_x = didi*N + W/2 * sin(alphar/2)* facW + Linlet
        c2_y = -Winlet2/2
        
        wp.MoveTo(c1_x, c1_y).Direction(-1,0)
        R2 = 3
        LL = c2_x - c1_x
        # close = Face(wp.Arc(R2+Winlet2,180).Line(LL).Arc(R2+Winlet2,180).Rotate(90).Line(Winlet2).Rotate(90).Arc(R2,-180).Line(LL).Arc(R2,-180).Close().Wire())
        
        close = Face(wp.Arc(R2+Winlet2,-180).Line(LL).Arc(R2+Winlet2,-180).Rotate(-90).Line(Winlet2).Rotate(-90).Arc(R2,180).Line(LL).Arc(R2,180).Close().Wire()).Reversed()
        # close.face.name

    teslavalve = vv
    # vv = onevalve
    if dim == 3:
        teslavalve = vv.Extrude( Winlet*Z )
        teslavalve.faces.name="wall"
        
    
        if closevalve == False:
            teslavalve.faces.Min(X).name="inlet"
            teslavalve.faces.Max(X).name="outlet"
            teslavalve.faces.Min(X).Identify(teslavalve.faces.Max(X),"periodic", IdentificationType.PERIODIC)

    else:
        teslavalve.edges.name="wall"
        teslavalve.faces.name="valve"
        teslavalve.edges.Min(X).name="inlet"
        teslavalve.edges.Max(X).name="outlet"
        
        if closevalve == True:
            close.edges.name = "wall"
            teslavalve =  Glue([teslavalve, close])
    
        if closevalve == False:        
            teslavalve.edges.Min(X).Identify(teslavalve.edges.Max(X),"periodic", IdentificationType.PERIODIC)

    return teslavalve

if __name__ == "__main__":
    ddim = 2
    valve = GetValve(N = 3, dim = ddim, R = 0.5, alpha = 25, Winlet = 1, beta = 180, L1 = 6.4, L2 = 7, Linlet = 5, closevalve=True)
    mesh = Mesh(OCCGeometry(valve, dim=ddim).GenerateMesh(maxh = 0.5))
    print(mesh.GetMaterials())
    print(mesh.GetBoundaries())
    mesh.Curve(3)

    Draw(mesh)