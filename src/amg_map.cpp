#define FILE_AMGMAP_CPP

#ifdef USE_TAU
#include "TAU.h"
#endif

#include "amg.hpp"


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



  /** ConcDMS **/

  ConcDMS :: ConcDMS (Array<shared_ptr<BaseDOFMapStep>> & _sub_steps)
    : BaseDOFMapStep(_sub_steps[0]->GetParDofs(), _sub_steps.Last()->GetMappedParDofs()),
      sub_steps(_sub_steps)
  {
    spvecs.SetSize(sub_steps.Size()-1);
    if(sub_steps.Size()>1) // TODO: test this out -> i think Range(1,0) is broken??
      for(auto k : Range(size_t(1),sub_steps.Size()))
	{ spvecs[k-1] = sub_steps[k]->CreateVector(); }
    vecs.SetSize(spvecs.Size());
    for (auto k : Range(spvecs.Size()))
      { vecs[k] = spvecs[k].get(); }
  }


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
  void ProlMap<TMAT> :: TransferF2C (const BaseVector * x_fine, BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ProlMap::TransferF2C", TAU_CT(*this), TAU_DEFAULT);
#endif

    RegionTimer rt(timer_hack_prol_f2c());
    x_fine->Distribute();
    prol_trans->Mult(*x_fine, *x_coarse);
    x_coarse->SetParallelStatus(DISTRIBUTED);
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

    RegionTimer rt(timer_hack_prol_c2f());
    x_coarse->Cumulate();
    prol->Mult(*x_coarse, *x_fine);
    x_fine->SetParallelStatus(CUMULATED);
  }


  template<class TMAT>
  void ProlMap<TMAT> :: AddC2F (double fac, BaseVector * x_fine, const BaseVector * x_coarse) const
  {
#ifdef USE_TAU
    TAU_PROFILE("ProlMap::AddC2F", TAU_CT(*this), TAU_DEFAULT);
#endif

    RegionTimer rt(timer_hack_prol_c2f());
    x_coarse->Cumulate(); x_fine->Cumulate();
    prol->MultAdd(fac, *x_coarse, *x_fine);
  }



  template<class TMAT>
  shared_ptr<BaseDOFMapStep> ProlMap<TMAT> :: Concatenate (shared_ptr<BaseDOFMapStep> other)
  {
    if (auto opmap = dynamic_pointer_cast<ProlMap<SPM_TM_C>>(other)) {
      auto comp_prol = MatMultAB<TMAT, SPM_TM_C> (*prol, *opmap->GetProl());
      auto comp_map = make_shared<ProlMap<mult_spm_tm<TMAT, SPM_TM_C>>> (comp_prol, GetParDofs(), opmap->GetMappedParDofs());
      return comp_map;
    }
    else
      { return nullptr; }
  }

  
  template<class TMAT>
  shared_ptr<BaseSparseMatrix> ProlMap<TMAT> :: AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const
  {

    auto tfmat = dynamic_pointer_cast<SPM_TM_F>(mat);
    if (tfmat == nullptr) {
      throw Exception(string("Cannot cast to ") + typeid(SPM_TM_F).name() + string(", type = ") + typeid(*mat).name() + string("!!") );
      return nullptr;
    }

    auto& self = const_cast<ProlMap<TMAT>&>(*this);
    self.prol_trans = TransposeSPM(*prol);
    self.prol = make_shared<SPM_P>(move(*prol));
    self.prol_trans = make_shared<trans_spm<SPM_P>>(move(*prol_trans));

    // cout << prol->Height() << " x " << prol->Width() << endl;
    // cout << tfmat->Height() << " x " << tfmat->Width() << endl;
    // print_tm_spmat(cout, *prol); cout << endl;
    
    auto spm_tm = RestrictMatrixTM<SPM_TM_F, TMAT> (*prol_trans, *tfmat, *prol);

    return make_shared<SPM_C>(move(*spm_tm));
  }

} // namespace amg

#include "amg_tcs.hpp"
