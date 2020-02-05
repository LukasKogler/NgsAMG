from ngsolve import *
from netgen.geom2d import unit_square
import sys

mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))

Vh1 = H1(mesh, dirichlet="left")

vecV = VectorH1(mesh, dirichlet="left")
# vecV = FESpace([Vh1, Vh1])
print("vecV free", sum(vecV.FreeDofs()), "of", vecV.ndof, "\n\n---------------------\n\n")
sys.stdout.flush()

roV = comp.Reorder(vecV)
print("vecVro free ", sum(roV.FreeDofs()), " of ", roV.ndof, "\n\n---------------------\n\n")
sys.stdout.flush()

compV = FESpace([vecV, vecV])
print("compV free", sum(compV.FreeDofs()), "of", compV.ndof, "\n\n---------------------\n\n")
sys.stdout.flush()

roC = comp.Reorder(compV)
print("roV free", sum(roC.FreeDofs()), "of", roC.ndof, "\n\n---------------------\n\n")
sys.stdout.flush()
