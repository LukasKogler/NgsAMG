#ifndef FILE_PYTHON_STOKES_HPP
#define FILE_PYTHON_STOKES_HPP


#include "python_amg.hpp"

namespace amg
{

template<class STOKES_PC, class LAM>
INLINE void ExportStokesAMGClass(py::module &m, string name, string desc, LAM lambda)
{
  ExportAMGClass<STOKES_PC, STOKES_PC> (m, name, desc, [&](auto & pyclass) {

    pyclass.def(py::init([&](shared_ptr<BilinearForm> bfa,
                              shared_ptr<BaseMatrix> wmat,
                              py::kwargs kwargs) {
        auto flags = CreateFlagsFromKwArgs(kwargs, py::none());
        return make_shared<STOKES_PC>(bfa, flags, "noname-pre", nullptr);
      }),
      py::arg("blf"),
      py::arg("weight_mat")=nullptr
    );

    pyclass.def(py::init([&](shared_ptr<FESpace> fes,
                             shared_ptr<BaseMatrix> A,
                             shared_ptr<BitArray> freeDofs,
                             shared_ptr<BaseMatrix> wmat,
                             bool finalize,
                             py::kwargs kwargs) {
        auto flags = CreateFlagsFromKwArgs(kwargs, py::none());
        auto pc = make_shared<STOKES_PC>(fes, flags, "noname-pre", nullptr, wmat);
        if (finalize)
        {
          pc->InitLevel(freeDofs);
          pc->FinalizeLevel(A);
        }
        return pc;
      }),
      py::arg("fes"),
      py::arg("mat"),
      py::arg("freedofs") = nullptr,
      py::arg("weight_mat") = nullptr,
      py::arg("finalize") = true
    );
    
    pyclass.def("GetNLoops", [&](shared_ptr<STOKES_PC> spc,
                                  int level,
                                  int rank) -> size_t {
        // rank 0 -> global N loops
        auto am = spc->GetAMGMatrix();
        auto map = am->GetMap();
        auto smoothers = am->GetSmoothers();
        size_t nloops = 0;
        if (level < smoothers.Size()) {
          auto hsm = unwrap_smoother<HiptMairSmoother>(smoothers[level]);
          if (hsm == nullptr)
            { throw Exception("No HiptMair smoother on level "+to_string(level)+", smoother = " + typeid(*smoothers[level]).name()); }
          auto & sm = const_cast<HiptMairSmoother&>(*hsm);
          auto C = sm.GetD();
          if (auto parC = dynamic_pointer_cast<ParallelMatrix>(C)) {
            auto pot_pds = parC->GetRowParallelDofs();
            if (rank == 0)
              { nloops = pot_pds->GetNDofGlobal(); }
            else if (rank == pot_pds->GetCommunicator().Rank())
              { nloops = pot_pds->GetNDofLocal(); }
          }
          else
            { nloops = C->Width(); }
        }
        nloops = map->GetUDofs(0).GetCommunicator().AllReduce(nloops, NG_MPI_MAX);
        return nloops;
      },
      py::arg("level") = 0,
      py::arg("rank") = 0
    );

    pyclass.def("GetLoop", [&](shared_ptr<STOKES_PC> spc,
                                shared_ptr<BaseVector> comp_vec,
                                int level,
                                int loop_num,
                                int rank,
                                bool print_gnum) {
        auto am = spc->GetAMGMatrix();
        auto map = am->GetMap();
        auto smoothers = am->GetSmoothers();
        shared_ptr<BaseVector> ran_vec = nullptr;
        if (level < smoothers.Size()) {
          auto hsm = unwrap_smoother<HiptMairSmoother>(smoothers[level]);
          if (hsm == nullptr)
            { throw Exception("no HiptMairSmoother on level" + to_string(level)); }
          auto & sm = const_cast<HiptMairSmoother&>(*hsm);
          // cout << " GetLoop " << level << ", loop " << loop_num << endl;
          auto C = sm.GetD();
          // cout << " C H/W = " << C->Height() << " " << C->Width() << endl;
          auto pot_vec = C->CreateRowVector();
          pot_vec.Distribute();
          ran_vec = C->CreateColVector();
          // cout << " pot_vec size = " << pot_vec.Size() << endl;
          // cout << " ran_vec size = " << ran_vec->Size() << endl;
          pot_vec.FVDouble() = 0.0;
          if (auto parC = dynamic_pointer_cast<ParallelMatrix>(C)) {
            auto pot_pds = parC->GetRowParallelDofs();
            auto pot_comm = pot_pds->GetCommunicator();
            int set_loc = -1;
            if ( (pot_comm.Size() > 2) && (rank == 0) ) { // need global enum!
              auto all = make_shared<BitArray>(pot_pds->GetNDofLocal());
              all->Set();
              Array<int> gdn; int gn;
              pot_pds->EnumerateGlobally(all, gdn, gn);
              for (auto k : Range(pot_pds->GetNDofLocal())) {
                if ( gdn[k] == loop_num ) { // why this ?? -> && pot_pds->IsMasterDof(k)) {
                  if (print_gnum)
                    { cout << " lev " << level << ", loc " << k << " -> glob " << loop_num << endl; }
                  if (pot_pds->IsMasterDof(k))
                  {
                    set_loc = k;
                  }
                  break;
                }
              }
            }
            else if ( ( (pot_comm.Rank() > 0) && (pot_comm.Rank() == rank) ) || // rank 0 == global!
                ( (pot_comm.Size() == 2) && (pot_comm.Rank() == 1) ) )
              { set_loc = loop_num; }
            if (set_loc != -1) {
              auto fv = pot_vec.FVDouble();
              if (print_gnum)
              {
                cout << " SET loop-FV " << level << " @ " << set_loc << "!" << endl;
              }
              if (set_loc > fv.Size())
                { throw Exception("Pot vec out of range on level " + to_string(level) + ", " +
                      to_string(fv.Size()) + "/" + to_string(set_loc)); }
              fv[set_loc] = 1.0;
            }
          }
          else { pot_vec.FVDouble()[loop_num] = 1.0; }
          // cout << " pot_vec " << pot_vec.Size() << endl;// << *pot_vec << endl;
          C->Mult(*pot_vec, *ran_vec);
        }
        // cout << " ran_vec " << ran_vec->Size() << endl;// << *ran_vec << endl;
        // cout << " trans " << level << " -> " << 0 << endl;
        // cout << " comp_vec " << comp_vec->Size() << endl;// << *comp_vec << endl;
        map->TransferAtoB(level, 0, ran_vec.get(), comp_vec.get());
      },
      py::arg("comp_vec"),
      py::arg("level"),
      py::arg("loop_num"),
      py::arg("rank") = 0,
      py::arg("print_gnum") = false
    );

    lambda(pyclass);
  });

} // ExportStokesAMGClass

} // namespace amg

#endif // FILE_PYTHON_STOKES_HPP