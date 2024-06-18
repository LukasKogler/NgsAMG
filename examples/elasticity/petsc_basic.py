from ngsolve import *
from netgen.geom2d import unit_square
import netgen.gui
import netgen.meshing as ngm
from mpi4py.MPI import COMM_WORLD

if COMM_WORLD.rank == 0:
    ngmesh = unit_square.GenerateMesh(maxh=0.1)
    for _ in range(4):
        ngmesh.Refine()
    mesh = Mesh(ngmesh.Distribute(COMM_WORLD))
else:
    mesh = Mesh(ngm.Mesh.Receive(COMM_WORLD))
order = 2
fes = H1(mesh, order=order, dirichlet="left|right|top|bottom")
print("Number of degrees of freedom: ", fes.ndof)
u,v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx)
f = LinearForm(fes)
f += 32 * (y*(1-y)+x*(1-x)) * v * dx
a.Assemble()
f.Assemble()


from ngsPETSc import KrylovSolver
solver = KrylovSolver(a, fes.FreeDofs(),
                      solverParameters={"ksp_type": "cg",
                                        "pc_type": "gamg"})
gfu = GridFunction(fes)
solver.solve(f.vec, gfu.vec)
exact = 16*x*(1-x)*y*(1-y)
print ("LU L2-error:", sqrt (Integrate ( (gfu-exact)*(gfu-exact), mesh)))
Draw(gfu)