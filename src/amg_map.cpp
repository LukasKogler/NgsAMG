
#define FILE_AMGMAP_CPP

#ifdef USE_TAU
#include "TAU.h"
#endif

#include "amg.hpp"
#include "amg_map.hpp"

#define AMG_EXTERN_TEMPLATES
#include "amg_tcs.hpp"
#undef AMG_EXTERN_TEMPLATES

namespace amg
{

  /** DOFMap **/

  Array<shared_ptr<BaseSparseMatrix>> DOFMap :: AssembleMatrices (shared_ptr<BaseSparseMatrix> finest_mat) const
  {
    static Timer t("DOFMap::AssembleMatrices");
    RegionTimer rt(t);
    shared_ptr<BaseSparseMatrix> last_mat = finest_mat;
    Array<shared_ptr<BaseSparseMatrix>> mats;
    mats.Append(last_mat);
    // cout << "SS IS : " << steps.Size() << endl;
    for (auto step_nr : Range(steps.Size())) {
      auto & step = steps[step_nr];
      auto next_mat = step->AssembleMatrix(last_mat);
      mats.Append(next_mat);
      last_mat = next_mat;
    }
    return mats;
  }


  void DOFMap :: TransferAtoB (int la, int lb, const BaseVector * vin, BaseVector * vout) const
  {
    if (la > lb)
      { ConcDMS(steps.Range(lb, la)).TransferC2F(vout, vin); }
    else if (la < lb)
      { ConcDMS(steps.Range(la, lb)).TransferF2C(vin, vout); }
    else
      { *vout = *vin; }
  } // DOFMap::TransferAtoB


  shared_ptr<DOFMap> DOFMap :: SubMap (int from, int to)
  {
    auto submap = make_shared<DOFMap>();
    int kmax = (to == -1) ? steps.Size() : to;
    for (auto k : Range(from, kmax))
      { submap->AddStep(steps[k]); }
    return submap;
  }

  /** END DofMap **/


  /** ConcDMS **/

  ConcDMS :: ConcDMS (FlatArray<shared_ptr<BaseDOFMapStep>> _sub_steps)
    : BaseDOFMapStep(_sub_steps[0]->GetParDofs(), _sub_steps.Last()->GetMappedParDofs()),
      sub_steps(_sub_steps.Size())
  {
    sub_steps = _sub_steps;
    spvecs.SetSize(sub_steps.Size()-1);
    if(sub_steps.Size()>1) // TODO: test this out -> i think Range(1,0) is broken??
      for(auto k : Range(size_t(1),sub_steps.Size()))
	{ spvecs[k-1] = sub_steps[k]->CreateVector(); }
    vecs.SetSize(spvecs.Size());
    for (auto k : Range(spvecs.Size()))
      { vecs[k] = spvecs[k].get(); }
  }


  void ConcDMS :: Finalize ()
  {
    for (auto step : sub_steps)
      { step->Finalize(); }
  } // ConcDMS::Finalize

  // bool ConcDMS :: CanPullBack (shared_ptr<BaseDOFMapStep> other)
  // {
  //   bool can_do_it = true;
  //   for (auto step : sub_steps)
  //     { can_do_it &= step->CanPullBack(other); }
  //   return can_do_it;
  // }


  // shared_ptr<BaseDOFMapStep> ConcDMS :: PullBack (shared_ptr<BaseDOFMapStep> other)
  // {
  //   if (!CanPullBack(other))
  //     { return nullptr; }
  //   shared_ptr<BaseDOFMapStep> out = other;
  //   for (int k = sub_steps.Size()-1; k >=0; k--)
  //     { out = sub_steps[k]->PullBack(out); }
  //   return out;
  // }


  void ConcDMS :: TransferF2C (const BaseVector * x_fine, BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ConcDMS::TransferF2C", TAU_CT(*this), TAU_DEFAULT);
#endif

    if (sub_steps.Size() == 1)
      { sub_steps[0]->TransferF2C(x_fine, x_coarse); }
    else {
      sub_steps[0]->TransferF2C(x_fine, vecs[0]);
      for (int l = 1; l<int(sub_steps.Size())-1; l++)
	sub_steps[l]->TransferF2C(vecs[l-1], vecs[l]);
      sub_steps.Last()->TransferF2C(vecs.Last(), x_coarse);
    }
  }


  void ConcDMS :: AddF2C (double fac, const BaseVector * x_fine, BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ConcDMS::AddF2C", TAU_CT(*this), TAU_DEFAULT);
#endif
    if (sub_steps.Size() == 1)
      { sub_steps[0]->AddF2C(fac, x_fine, x_coarse); }
    else {
      sub_steps[0]->TransferF2C(x_fine, vecs[0]);
      for (int l = 1; l<int(sub_steps.Size())-1; l++)
	sub_steps[l]->TransferF2C(vecs[l-1], vecs[l]);
      sub_steps.Last()->AddF2C(fac, vecs.Last(), x_coarse);
    }
  }


  void ConcDMS :: TransferC2F (BaseVector * x_fine, const BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ConcDMS::TransferC2F", TAU_CT(*this), TAU_DEFAULT);
#endif

    // cout << "conc map transC2F " << endl;
    // for (auto step : sub_steps)
    //   { cout << typeid(*step).name() << endl; }

    if (sub_steps.Size() == 1)
      { sub_steps[0]->TransferC2F(x_fine, x_coarse); }
    else {
      sub_steps.Last()->TransferC2F(vecs.Last(), x_coarse);
      for (int l = sub_steps.Size()-2; l>0; l--)
	sub_steps[l]->TransferC2F(vecs[l-1], vecs[l]);
      sub_steps[0]->TransferC2F(x_fine, vecs[0]);
    }
  }


  void ConcDMS :: AddC2F (double fac, BaseVector * x_fine, const BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ConcDMS::AddC2F", TAU_CT(*this), TAU_DEFAULT);
#endif

    // cout << "conc map addc2f " << endl;
    // for (auto step : sub_steps)
    //   { cout << typeid(*step).name() << endl; }

    if (sub_steps.Size() == 1)
      { sub_steps[0]->AddC2F(fac, x_fine, x_coarse); }
    else {
      sub_steps.Last()->TransferC2F(vecs.Last(), x_coarse);
      for (int l = sub_steps.Size()-2; l>0; l--)
	sub_steps[l]->TransferC2F(vecs[l-1], vecs[l]);
      sub_steps[0]->AddC2F(fac, x_fine, vecs[0]);
    }
  }


  shared_ptr<BaseSparseMatrix> ConcDMS :: AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const
  {
    shared_ptr<BaseSparseMatrix> cmat = mat;
    for (auto& step : sub_steps)
      cmat = step->AssembleMatrix(cmat);
    return cmat;
  }


  /** ProlMap **/

  template<class TMAT>
  void ProlMap<TMAT> :: Finalize ()
  {
    this->BuildPT();
    if (dynamic_pointer_cast<SPM_P>(prol) == nullptr)
      { prol = make_shared<SPM_P>(move(*prol)); }
    if (dynamic_pointer_cast<trans_spm<SPM_P>>(prol_trans) == nullptr)
      { prol_trans = make_shared<trans_spm<SPM_P>>(move(*prol_trans)); }
  } // ProlMap::Finalize


  template<class TMAT>
  void ProlMap<TMAT> :: TransferF2C (const BaseVector * x_fine, BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ProlMap::TransferF2C", TAU_CT(*this), TAU_DEFAULT);
#endif

    // cout << " prol f2c " << endl;
    // cout << " f c lens " << x_fine->Size() << " " << x_coarse->Size() << endl;
    // cout << " prolt dims " << prol_trans->Height() << " x " << prol_trans->Width() << endl;

    RegionTimer rt(timer_hack_prol_f2c());
    x_fine->Distribute();
    // cout << " prol f2c " << endl;
    prol_trans->Mult(*x_fine, *x_coarse);
    // cout << " prol f2c " << endl;
    x_coarse->SetParallelStatus(DISTRIBUTED);

    // cout << " prol f2c done " << endl;

    // cout << "fine: " << endl << *x_fine << endl;
    // cout << "coarse: " << endl << *x_coarse << endl;
  }


  template<class TMAT>
  void ProlMap<TMAT> :: AddF2C (double fac, const BaseVector * x_fine, BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ProlMap::AddF2C", TAU_CT(*this), TAU_DEFAULT);
#endif

    RegionTimer rt(timer_hack_prol_f2c());
    x_fine->Distribute(); x_coarse->Distribute();
    prol_trans->MultAdd(fac, *x_fine, *x_coarse);
  }

  template<class TMAT>
  void ProlMap<TMAT> :: TransferC2F (BaseVector * x_fine, const BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ProlMap::TransferC2F", TAU_CT(*this), TAU_DEFAULT);
#endif

    // cout << "prol trans c2f " << prol->Height() << " x " << prol->Width() << endl;

    // cout << endl;
    // print_tm_spmat(cout, *prol);
    // cout << endl;

    // cout << "x caorse " << endl; prow2(x_coarse->FVDouble()); cout << endl;

    RegionTimer rt(timer_hack_prol_c2f());
    x_coarse->Cumulate();
    prol->Mult(*x_coarse, *x_fine);
    x_fine->SetParallelStatus(CUMULATED);

    // cout << "x fine " << endl; prow2(x_fine->FVDouble()); cout << endl;
  }


  template<class TMAT>
  void ProlMap<TMAT> :: AddC2F (double fac, BaseVector * x_fine, const BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ProlMap::AddC2F", TAU_CT(*this), TAU_DEFAULT);
#endif

    // cout << "prol add c2f" << endl;

    // cout << endl;
    // print_tm_spmat(cout, *prol);
    // cout << endl;

    // cout << "x coarse/fine sizes " << endl << x_coarse->Size() << " " << x_fine->Size() << endl;
    // cout << " prol dims " << prol->Height() << " x " << prol->Width() << endl;
    // cout << "x caorse " << endl << x_coarse->FVDouble() << endl;

    // cout << "x caorse " << endl << x_coarse->FVDouble() << endl;

    RegionTimer rt(timer_hack_prol_c2f());
    x_coarse->Cumulate(); x_fine->Cumulate();
    prol->MultAdd(fac, *x_coarse, *x_fine);

    // cout << "x fine " << endl << x_fine->FVDouble() << endl;
  }



  template<class TMAT>
  shared_ptr<BaseDOFMapStep> ProlMap<TMAT> :: Concatenate (shared_ptr<BaseDOFMapStep> other)
  {
    if (auto opmap = dynamic_pointer_cast<ProlMap<SPM_TM_C>>(other)) {

      // cout << "conc prols " << endl;
      // cout << " left " << GetProl() << endl;
      // cout << " left:  " << GetProl()->Height() << " x " << GetProl()->Width() << endl;
      // cout << " left typeid " << typeid(*this).name() << endl;
      // print_tm_spmat(cout, *prol); cout << endl<< endl;
      // cout << " right " << other << " " << opmap << endl;
      // cout << " right mat: " << opmap->GetProl() << endl;
      // cout << " right: " << opmap->GetProl()->Height() << " x " << opmap->GetProl()->Width() << endl;
      // print_tm_spmat(cout, *opmap->GetProl()); cout << endl<< endl;

      auto comp_prol = MatMultAB<TMAT, SPM_TM_C> (*prol, *opmap->GetProl());

      // cout << " comp: "  << comp_prol->Height() << " x " << comp_prol->Width() << endl;
      // print_tm_spmat(cout, *comp_prol); cout << endl<< endl;

      auto comp_map = make_shared<ProlMap<mult_spm_tm<TMAT, SPM_TM_C>>> (comp_prol, GetParDofs(), opmap->GetMappedParDofs());
      return comp_map;
    }
    else
      { return nullptr; }
  }

  
  template<class TMAT>
  shared_ptr<trans_spm_tm<TMAT>> ProlMap<TMAT> :: GetProlTrans () const
  {
    if (prol_trans == nullptr)
      { const_cast<ProlMap<TMAT>&>(*this).BuildPT(); }
    return prol_trans;
  }

  template<class TMAT>
  void ProlMap<TMAT> :: BuildPT (bool force)
  {
    if (force || (prol_trans == nullptr))
      { prol_trans = TransposeSPM(*prol); }
  }

  template<class TMAT>
  shared_ptr<BaseSparseMatrix> ProlMap<TMAT> :: AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const
  {

    // cout << "prolmap assmat, type " << typeid(*this).name() << endl;
    // cout << "prol type " << typeid(*prol).name() << endl;
    // cout << "prol dims " << prol->Height() << " x " << prol->Width() << endl;
    // cout << "prol, fmat " << prol << ", " << mat << endl;
    // cout << "fmat type " << typeid(*mat).name() << endl;
    // cout << "fmat dims " << mat->Height() << " x " << mat->Width() << endl;

    auto tfmat = dynamic_pointer_cast<SPM_TM_F>(mat);
    if (tfmat == nullptr) {
      throw Exception(string("Cannot cast to ") + typeid(SPM_TM_F).name() + string(", type = ") + typeid(*mat).name() + string("!!") );
      return nullptr;
    }

    auto& self = const_cast<ProlMap<TMAT>&>(*this);
    self.Finalize();
    
    // if (prol->Width() < 100) {
      // cout << " fmat: " << endl; print_tm_spmat(cout, *tfmat); cout << endl<< endl;
      // cout << " prol: " << endl; print_tm_spmat(cout, *prol); cout << endl<< endl;
    // }

    // auto spm_tm = RestrictMatrixTM<SPM_TM_F, TMAT> (*prol_trans, *tfmat, *prol);
    shared_ptr<SPM_TM_C> spm_tm = RestrictMatrixTM<SPM_TM_F, TMAT> (*prol_trans, *tfmat, *prol);

    // if (prol->Width() < 100)
      // { cout << " cmat: " << endl; print_tm_spmat(cout, *spm_tm); cout << endl<< endl; }

    return make_shared<SPM_C>(move(*spm_tm));
  }


  /** MultiDofMapStep **/

  MultiDofMapStep :: MultiDofMapStep (FlatArray<shared_ptr<BaseDOFMapStep>> _maps)
    : BaseDOFMapStep(_maps[0]->GetParDofs(), _maps[0]->GetMappedParDofs())
  {
    maps.SetSize(_maps.Size());
    maps = _maps;
  }


  void MultiDofMapStep :: Finalize ()
  {
    for (auto map : maps)
      { map->Finalize(); }
  } // MultiDofMapStep::Finalize


  void MultiDofMapStep :: TransferF2C (const BaseVector * x_fine, BaseVector * x_coarse) const
  { GetPrimMap()->TransferF2C(x_fine, x_coarse); } // MultiDofMapStep :: TransferF2C


  void MultiDofMapStep :: AddF2C (double fac, const BaseVector * x_fine, BaseVector * x_coarse) const
  { 
    GetPrimMap()->AddF2C(fac, x_fine, x_coarse);
  } // MultiDofMapStep :: AddF2C


  void MultiDofMapStep :: TransferC2F (BaseVector * x_fine, const BaseVector * x_coarse) const
  {
    GetPrimMap()->TransferC2F(x_fine, x_coarse);
  } // MultiDofMapStep :: TransferC2F


  void MultiDofMapStep :: AddC2F (double fac, BaseVector * x_fine, const BaseVector * x_coarse) const
  {
    GetPrimMap()->AddC2F(fac, x_fine, x_coarse);
  } // MultiDofMapStep :: AddC2F


  bool MultiDofMapStep :: CanConcatenate (shared_ptr<BaseDOFMapStep> other)
  {
    if (auto mdms = dynamic_pointer_cast<MultiDofMapStep>(other)) {
      bool cc = mdms->GetNMaps() == GetNMaps();
      for (auto k : Range(maps))
	{ cc &= GetMap(k)->CanConcatenate(mdms->GetMap(k)); }
      return cc;
    }
    else if (GetNMaps() == 1)
      { return maps[0]->CanConcatenate(other); }
    else
      { return false; }
  } // MultiDofMapStep :: CanConcatenate


  shared_ptr<BaseDOFMapStep> MultiDofMapStep :: Concatenate (shared_ptr<BaseDOFMapStep> other)
  {
    if (!CanConcatenate(other))
      { return nullptr; }
    if (auto mdms = dynamic_pointer_cast<MultiDofMapStep>(other)) {
      Array<shared_ptr<BaseDOFMapStep>> cmaps(GetNMaps());
      for (auto k : Range(cmaps))
	{ cmaps[k] = GetMap(k)->Concatenate(mdms->GetMap(k)); }
      return make_shared<MultiDofMapStep>(cmaps);
    }
    else if (GetNMaps() == 1)
      { return maps[0]->Concatenate(other); }
    return nullptr;
  } // MultiDofMapStep :: Concatenate


  shared_ptr<BaseDOFMapStep> MultiDofMapStep :: PullBack (shared_ptr<BaseDOFMapStep> other)
  {
    if (auto mdms = dynamic_pointer_cast<MultiDofMapStep>(other)) {
      Array<shared_ptr<BaseDOFMapStep>> cmaps(GetNMaps());
      for (auto k : Range(cmaps)) {
	  cmaps[k] = GetMap(k)->PullBack(mdms->GetMap(k));
	if (cmaps[k] == nullptr)
	  { throw Exception("MultiDofMapStep could not pull back a component!"); }
      }
      return make_shared<MultiDofMapStep>(cmaps);
    }
    else if (GetNMaps() == 1)
      { return maps[0]->PullBack(other); }
    return maps[0]->PullBack(other);
  } // MultiDofMapStep :: PullBack


  shared_ptr<BaseSparseMatrix> MultiDofMapStep :: AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const
  {
    return maps[0]->AssembleMatrix(mat);
  } // MultiDofMapStep :: AssembleMatrix

  /** END MultiDofMapStep **/

  shared_ptr<BaseDOFMapStep> MakeSingleStep2 (FlatArray<shared_ptr<BaseDOFMapStep>> init_steps)
  {
    const int iss = init_steps.Size();
    if (iss == 0)
      { return nullptr; }
    Array<shared_ptr<BaseDOFMapStep>> sub_steps(iss); sub_steps.SetSize0();
    for (int k = 0; k < iss; k++) {
      shared_ptr<BaseDOFMapStep> conc_step = init_steps[k];
      int j = k+1;
      for ( ; j < iss; j++)
	if (auto x = conc_step->Concatenate(init_steps[j]))
	  { conc_step = x; k++; }
	else
	  { break; }
      sub_steps.Append(conc_step);
    }
    shared_ptr<BaseDOFMapStep> final_step = nullptr;
    if (sub_steps.Size() == 1)
      { return sub_steps[0]; }
    else
      { return make_shared<ConcDMS>(sub_steps); }
  } // MakeSingleStep2


  shared_ptr<BaseDOFMapStep> MakeSingleStep (FlatArray<shared_ptr<BaseDOFMapStep>> sub_steps)
  {
    int nmaps = 0, nss = sub_steps.Size();

    if (nss == 0)
      { return nullptr; }

    for (int k = 0; (k < nss) & (nmaps != -1); k++) {
      if (auto mdms = dynamic_pointer_cast<MultiDofMapStep>(sub_steps[k])) {
	if (nmaps == 0)
	  { nmaps = mdms->GetNMaps(); }
	else
	  { nmaps = (mdms->GetNMaps() == nmaps) ? nmaps : -1; }
      }
      else
	{ nmaps = -1; }
    }
    if (nmaps != -1) {
      Array<shared_ptr<BaseDOFMapStep>> multi_subs(nmaps), jmaps(nss);
      for (auto k : Range(nmaps)) {
	for (auto j : Range(nss))
	  { jmaps[j] = dynamic_pointer_cast<MultiDofMapStep>(sub_steps[j])->GetMap(k); }
	multi_subs[k] =  MakeSingleStep2(jmaps);
      }
      return make_shared<MultiDofMapStep>(multi_subs);
    }
    else
      { return MakeSingleStep2(sub_steps); }
  } // MakeSingleStep

} // namespace amg

#include "amg_tcs.hpp"

// namespace amg
// {
//   template class ProlMap<SparseMatrixTM<double>>;
//   template class ProlMap<SparseMatrixTM<Mat<2,2,double>>>;
// } // namespace amg
