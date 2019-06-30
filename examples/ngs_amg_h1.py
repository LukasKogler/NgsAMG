from ngsolve import *
import netgen, ngs_amg

comm = mpi_world

def gen_mesh(geo, maxh, nref):
    ngsglobals.msg_level = 0
    if comm.rank==0:
        ngm = geo.GenerateMesh(maxh=maxh)
        if comm.size > 0:
            ngm.Distribute(comm)
    else:
        ngm = netgen.meshing.Mesh.Receive(comm)
        ngm.SetGeometry(geo)
    for l in range(nref):
        ngm.Refine()
    return Mesh(ngm)

maxh = 0.03
nref = 2
order = 1
condense = True
do_petsc = True

mesh = gen_mesh(netgen.csg.unit_cube, maxh, nref)

ngsglobals.msg_level = 1

V = H1(mesh, order=order, dirichlet='left|right|top')

u,v = V.TnT()
a = BilinearForm(V, symmetric=False, condense = condense)
a += SymbolicBFI(grad(u)*grad(v))

c = Preconditioner(a, "ngs_amg.h1_scal",
                   ngs_amg_log_level = 2,
                   ngs_amg_log_file = "")

f = LinearForm(V)
f += SymbolicLFI(v)
gfu = GridFunction(V)


# paje_size = 100 * 1024 * 1024 if comm.rank in [0,1,comm.size-1] else 0
paje_size = 0
with TaskManager(pajetrace = paje_size):
    a.Assemble()
    f.Assemble()

    ## Ngs-AMG + NGs-side CG

    comm.Barrier()
    t1 = -comm.WTime()
    if comm.rank == 0:
        print('------- Ngs-AMG + NGs-side CG --------')
    solvers.CG(mat=a.mat, pre=c, sol=gfu.vec, rhs=f.vec, tol=1e-6, maxsteps=100, printrates=comm.rank==0)
    if comm.rank == 0:
        print('---------------')
    comm.Barrier()
    t1 = t1 + comm.WTime()
    
    if do_petsc:
        ## NGs-side side AMG, PETSc side CG
        import ngs_petsc as petsc

        awrap = petsc.FlatPETScMatrix(a.mat, V.FreeDofs(condense))
        cwrap = petsc.NGs2PETScPrecond(pc=c, mat=awrap)
        ksp = petsc.KSP(awrap, finalize=False,
                        petsc_options = {"ksp_type" : "cg",
                                         "ksp_monitor" : "",
                                         "ksp_rtol" : "1e-6",
                                         "ksp_converged_reason" : ""})

        ## Ngs-AMG + PETSc-side CG

        ksp.SetPC(cwrap)
        ksp.Finalize()
        comm.Barrier()
        t1ps = -comm.WTime()
        if comm.rank == 0:
            print('------ Ngs-AMG + PETSc-side CG ---------')
        gfu.vec.data = ksp * f.vec
        if comm.rank == 0:
            print('---------------')
        comm.Barrier()
        t1ps = t1ps + comm.WTime()


        ## PETSc-side AMG + NGs-side CG

        p_a = petsc.PETScMatrix(a.mat, V.FreeDofs())
        # gfu.Set(1)
        # p_a.SetNearNullSpace([gfu.vec])
        c2 = petsc.PETSc2NGsPrecond(p_a, petsc_options = { "pc_type" : "gamg"})
        comm.Barrier()
        t2 = -comm.WTime()
        if comm.rank == 0:
            print('------- PETSc-side AMG + NGs-side CG --------')
        solvers.CG(mat=a.mat, pre=c2, sol=gfu.vec, rhs=f.vec, tol=1e-6, maxsteps=100, printrates=comm.rank==0)
        if comm.rank == 0:
            print('---------------')
        comm.Barrier()
        t2 = t2 + comm.WTime()


        ## PETSc-side AMG + PETSc-side CG

        ksp.SetPC(c2)
        ksp.Finalize()

        comm.Barrier()
        t2ps = -comm.WTime()
        if comm.rank == 0:
            print('------ PETSc-side AMG + PETSc-side CG ---------')
        gfu.vec.data = ksp * f.vec
        if comm.rank == 0:
            print('---------------')
        comm.Barrier()
        t2ps = t2ps + comm.WTime()
            
    if comm.rank == 0:
        print('\n ----------- ')
        print('ndof : ', V.ndofglobal)
        print('low order ndof: ', V.lospace.ndofglobal)
        print(' --- NGs-side AMG, NGs-side CG --- ')
        print('t solve', t1)
        print('dofs / (sec * np) ', V.ndofglobal / (t1 * max(comm.size-1, 1)) )
        if do_petsc:
            print(' --- NGs-side AMG, PETSc-side CG --- ')
            print('t solve', t1ps)
            print('dofs / (sec * np) ', V.ndofglobal / (t1ps * max(comm.size-1, 1)) )
            print('--- PETSc-side AMG, NGs-side CG --- ')
            print('t solve', t2)
            print('dofs / (sec * np) ', V.ndofglobal / (t2 * max(comm.size-1, 1)) )
            print('--- PETSc-side AMG, PETSc-side CG --- ')
            print('t solve', t2ps)
            print('dofs / (sec * np) ', V.ndofglobal / (t2ps * max(comm.size-1, 1)) )
        print(' ----------- \n')


