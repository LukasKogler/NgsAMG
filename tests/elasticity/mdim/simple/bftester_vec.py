import sys
from ngsolve import *

def get_defos(dim, gf_d, gf_r):
    if dim == 2:
        cd_gf_r = CoeffciientFunction( (gf_r[0], gf_r[1]) )
    else:
        cd_gf_r = CoeffciientFunction( (gf_r[0], gf_r[1], gf_r[2]) )
    gf_d.Set(cf_gf_r)

def get_defos2(dim, gf_d, gf_r):
    gf_d.vec.SetParallelStatus(gf_r.vec.GetParallelStatus())
    vd = gf_d.vec.local_vec
    vr = gf_r.vec.local_vec
    for k in range(len(vr)):
        for l in range(dim):
            vd[k][l] = vr[k][l]
    sys.stdout.flush()
    print('get_defos: ', len(vr), len(vd))
    for k in range(len(vr)):
        print(k, ', defo = ', [x for x in vd[k]], ", rot = ", [ x for x in vr[k]])
    sys.stdout.flush()

def dummy():
    pass
            
def shape_test(mesh, maxh, V, a, pc, multidim, order=1):
    dim = mesh.dim

    test_bf = GridFunction(V)
    comm = mpi_world
    bf_vec = test_bf.vec.CreateVector()

    ssol = Draw(test_bf, name="BFsol")
    if V.dim == dim:
        defo_bf = test_bf
        update_vecs = lambda : dummy()
    else:
        V2 = H1(mesh, order=order, dim=dim)
        defo_bf = GridFunction(V2)
        update_vecs = lambda : get_defos2(dim, defo_bf, test_bf)
    smesh = Draw(mesh, deformation=defo_bf, name="BFmesh")
    # smesh = Draw(mesh, deformation=test_bf, name="BFmesh")
    if smesh is not None and hasattr(smesh, "sub_scenes"):
        for ss in smesh.sub_scenes:
            wid = ss._parameters["Show"][1]._sub_parameters[0]._widget
            case = 3
            if case==1:
                wid.setChecked("outer", False)
                wid.setChecked("left", False)
                wid.setChecked(0,"back")
            elif case==2:
                wid.setChecked("left", False)
                wid.setChecked("front", False)
                wid.setChecked("right", False)
                wid.setChecked("bottom", False)
                wid.setChecked("top", False)
    def set_defo_scale (val):
        if smesh is not None:
            smesh.setDeformation(True)
            smesh.setDeformationScale(val)

    energy = GridFunction(V)
    W = L2(mesh, order=1)
    absenergy = GridFunction(W)
    sc_energy = Draw(absenergy, name="energy")

    nlevs = pc.GetNLevels(1)
    stop = 0
    lev,dof = (0,0)
    mode = 1
    max_l = pc.GetNLevels(1 if comm.size>1 else 0)-1
    max_d = [pc.GetNDof(k,0) for k in range(max_l+1)]
    lev,dof,add = 0,0,0
    scal = 1.0 if comm.rank==0 else 0
    while True:
        if comm.rank==0:
            print('\nmax level is: ', max_l)
            print('ND on levels: ', [str(x)+':'+str(y) for x,y in enumerate(max_d) ])
            print('NV on levels: ', [str(x)+':'+str(int(y/multidim)) for x,y in enumerate(max_d) ])
            ret = input('level, dof to draw next? (Q to quit)\n')
            if len(ret)>0 and ret[0]=='a':
                ret = ret[1:]
                add = 1
            else:
                add = 0;
            if not len(ret):
                dof = dof+1
            elif ret == 'bf':
                mode = 1
                dof = 0
                print('switched to base-functions')
            elif ret in ['q','Q']:
                stop = 1
            elif ret[0] in ['l', 'L']:
                lev = int(ret[1:])
                dof = 0
            elif ret[0] == 's':
                set_defo_scale(float(ret[1:]))
                continue
            elif ret[0] == 'm':
                mode = int(ret[1])
                continue
            elif len(ret.split(','))==1:
                try:
                    olddof = dof
                    dof = int(ret)
                except:
                    dof = olddof
                    continue
            elif len(ret.split(','))==2:
                lev, dof = [int(x) for x in ret.split(',')]
                scal = 1.0
            else:
                rs = ret.split(',')
                lev = int(rs[0])
                dof = int(rs[1])
                scal = float(rs[2])
        else:
            lev,dof,add,mode = 0,0,0,0
            scal = 0.0
        if comm.rank==0:
            print(lev, dof, add, scal, mode)
        comm.Barrier()
        stop = comm.Sum(stop)
        if stop:
            break
        add = comm.Sum(add)
        lev = comm.Sum(lev)
        dof = comm.Sum(dof)
        scal = comm.Sum(scal)
        mode = comm.Sum(mode)
        #test_bf.vec[:] = 0
        if mode == 1:
            if comm.rank==0:
                print('bf', dof, 'from level', lev)
                print('add',add,'scal',scal)
            pc.GetBF(bf_vec, lev, 0, dof)
            if add==0:
                test_bf.vec.data = scal * bf_vec
            else:
                test_bf.vec.data += scal * bf_vec
            print('vec: ', [(k, test_bf.vec[k]) for k in range(len(test_bf.vec))])
        elif mode == 2:
            if comm.rank==0:
                print('sum up, start with ', dof, 'from level', lev)
            test_bf.vec[:] = 0
            for dnr in [dof + multidim * l for l in range(int(max_d[lev]/multidim))]:
                pc.GetBF(bf_vec, lev, 0, dnr)
                test_bf.vec.data += bf_vec
            print('vec', test_bf.vec)
        else:
            print('unknown mode!')
        energy.vec.data = a.mat * test_bf.vec
        absenergy.Set(InnerProduct(energy, energy))
        normbf_e = InnerProduct(test_bf.vec,energy.vec)
        normbf_l2 = Norm(test_bf.vec) * maxh**dim #right scaling
        if comm.rank==0:
            print('BF energy norm:      ', normbf_e)
            print('BF L2 norm:          ', normbf_l2)
            print('BF rel. energy norm: ', normbf_e/normbf_l2 if normbf_l2!=0 else -1)
        update_vecs()
        Redraw()
