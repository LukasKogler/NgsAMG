import mpi4py.MPI as MPI

from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg
from NgsAMG import *

from smoother_utils import MakeFacetBlocks,TestSmoother

# set up HDiv-HDG Stokes as a test case
order = 3
if MPI.COMM_WORLD.rank == 0:
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.1).Distribute(MPI.COMM_WORLD))
else:
    mesh = Mesh(netgen.meshing.Mesh.Receive(MPI.COMM_WORLD))
V1 = HDiv ( mesh, order = order, dirichlet = ".*")
V2 = TangentialFacetFESpace(mesh, order = order, dirichlet = ".*", highest_order_dc=True )
V = V1 * V2
(u,uhat),(v,vhat) = V.TnT()
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size
def tang(vec):
    return vec - (vec*n)*n
alpha = 4  # SIP parameter
nu = 1
dS = dx(element_boundary=True)
a = BilinearForm ( V, eliminate_internal=True )
a += nu * InnerProduct ( Grad(u), Grad(v) ) * dx
a += nu * InnerProduct ( Grad(u)*n, tang(vhat-v) ) * dS
a += nu * InnerProduct ( Grad(v)*n, tang(uhat-u) ) * dS
a += nu * alpha*order*order/h * InnerProduct(tang(vhat-v), tang(uhat-u)) * dS

a.Assemble()

globCoupling = MPI.COMM_WORLD.allreduce(sum(V.FreeDofs(True)), op=MPI.SUM)
if MPI.COMM_WORLD.rank == 0:
    print(f"#coupling DOFS = {globCoupling}")

blocks = MakeFacetBlocks(V, V.FreeDofs(True))

if MPI.COMM_WORLD.size == 1:
    # NGSolve built-in smoothers
    gs = a.mat.CreateSmoother(V.FreeDofs(True))
    bgs = a.mat.CreateBlockSmoother(blocks)

# NgsAMG hybrid smoothers - MPI-parallel & communication overlapping
hybGS = NgsAMG.CreateHybridGSS(mat=a.mat,freedofs=V.FreeDofs(True))
hybBGS = NgsAMG.CreateHybridBlockGSS(mat=a.mat,blocks=blocks)

# shared-memory parallel version (slower than "normal" block-GS usually)
# hybBGS = NgsAMG.CreateHybridBlockGSS(mat=a.mat,blocks=blocks,sm2=False,shm=True)

if MPI.COMM_WORLD.size == 1:
    TestSmoother(gs, a.mat, True, "NgSolve-GS")
TestSmoother(hybGS, a.mat, True, "NgsAMG-GS")

if MPI.COMM_WORLD.size == 1:
    TestSmoother(bgs, a.mat, True,    "NgSolve-Block-GS")
TestSmoother(hybBGS, a.mat, True, "NgsAMG-Block-GS")

