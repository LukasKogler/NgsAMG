from ngsolve import *
import mpi4py.MPI as MPI

def MakeFacetBlocks(V, freedofs=None):
    blocks = []
    if freedofs is not None:
        for facet in V.mesh.facets:
            block = list( dof for dof in V.GetDofNrs(facet) if dof>=0 and freedofs[dof])
            if len(block):
                blocks.append(block)
    else:
        for facet in V.mesh.facets:
            block = list( dof for dof in V.GetDofNrs(facet) if dof>=0)
            if len(block):
                blocks.append(block)

    totSize   = MPI.COMM_WORLD.allreduce(sum(len(x) for x in blocks), op=MPI.SUM)
    totBlocks = MPI.COMM_WORLD.allreduce(len(blocks), op=MPI.SUM)
    avgSize =  totSize / totBlocks

    if MPI.COMM_WORLD.rank == 0:
        print(f"(globally) created {totBlocks} facet-blocks of average size {avgSize}")
    return blocks

class SmootherAsPrecond (BaseMatrix):
    def __init__(self, smoother, mat, ngsSmoother=True):
        super(SmootherAsPrecond, self).__init__()
        self.ngsSmoother = ngsSmoother # smooth with residuum
        self.A = mat
        self.S = smoother
        self.res = self.S.CreateColVector()
    def IsComplex(self):
        return False
    def Height(self):
        return self.S.height
    def Width(self):
        return self.S.width
    def CreateColVector(self):
        return self.S.CreateColVector()
    def CreateRowVector(self):
        return self.S.CreateRowVector()
    def MultAdd(self, scal, b, x):
        self.Mult(b, self.xtemp)
        x.data += scal * self.xtemp
    def MultTransAdd(self, scal, b, x):
        self.MultAdd(scal, b, x)
    def MultTrans(self, b, x):
        self.Mult(b, x)
    def Mult(self, b, x):
        x[:] = 0.0
        if not self.ngsSmoother:
            # update residual with forward smooth
            self.res.data = b
            self.S.Smooth(x, b, self.res, x_zero=True, res_updated=True, update_res=True)
            self.S.SmoothBack(x, b, self.res, x_zero=False, res_updated=True, update_res=False)
        else:
            self.S.Smooth(x, b)
            self.S.SmoothBack(x, b)

def TestSmoother(sm, a, isNGS, title):

    N = 20
    asPre = SmootherAsPrecond(sm, a, ngsSmoother=isNGS)
    lams = list(la.EigenValues_Preconditioner(mat=a, pre=asPre, tol=1e-6))

    # ANY vectors
    u = a.CreateVector()
    u[:] = 2.0
    f = a.CreateVector()
    f[:] = 1.0

    tf = Timer("smooth - FW")
    MPI.COMM_WORLD.Barrier()
    tf.Start()
    for k in range(N):
        sm.Smooth(u, f)
    MPI.COMM_WORLD.Barrier()
    tf.Stop()

    tb = Timer("smooth - BW")
    MPI.COMM_WORLD.Barrier()
    tb.Start()
    for k in range(N):
        sm.SmoothBack(u, f)
    MPI.COMM_WORLD.Barrier()
    tb.Stop()

    if MPI.COMM_WORLD.rank == 0:
        print(f"\nTesting smoother {title}:")
        print(f"  If used as preconditioner:")
        print(f"      lam min:   {lams[0]}")
        print(f"      lam max:   {lams[-1]}")
        print(f"      condition: {lams[-1] / lams[0]}")
        print(f"  sec per smooth forward:   {tf.time/N}")
        print(f"  sec per smooth backward:  {tb.time/N}")


def TestSPMV(a, title):

    N = 20
    # ANY vectors
    u = a.CreateVector()
    u[:] = 2.0
    f = a.CreateVector()
    f[:] = 1.0

    tf = Timer("mult")
    MPI.COMM_WORLD.Barrier()
    tf.Start()
    for k in range(N):
        u.data = a * f
    MPI.COMM_WORLD.Barrier()
    tf.Stop()

    if MPI.COMM_WORLD.rank == 0:
        print(f"\nTesting SPMV {title}:")
        print(f"  sec per spmv:   {tf.time/N}")
