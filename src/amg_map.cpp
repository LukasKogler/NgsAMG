#define FILE_AMGMAP_CPP

#include "amg.hpp"

namespace amg
{

  void DOFMap :: Finalize (FlatArray<size_t> acutoffs, shared_ptr<BaseDOFMapStep> embed_step)
  {
    cutoffs.SetSize(acutoffs.Size());
    cutoffs = acutoffs;
    ConcSteps();
    // cout << "AFTER CONC" << endl;
    // for (auto step : steps)
    //   { cout << typeid(*step).name()  << endl; }
    // cout << "EMBED " << embed_step << endl;
    if (embed_step != nullptr) {
      // cout << "EMBED : " << typeid(*embed_step).name() << endl;
      auto conc = dynamic_pointer_cast<ConcDMS>(steps[0]);
      shared_ptr<BaseDOFMapStep> fstep = conc ? conc->hacky_get_first_step() : steps[0];
      fstep = embed_step->Concatenate(fstep);
      if (conc) conc->hacky_replace_first_step(fstep);
      else steps[0] = fstep;
    }
  }
    
  void DOFMap :: ConcSteps ()
  {
    // cout << "CONCSTEPS: " << endl; prow2(steps); cout << endl;
    // for (auto step : steps)
    //   { cout << typeid(*step).name()  << endl; }
    static Timer t("DOFMap::ConcSteps");
    RegionTimer rt(t);

    if (steps.Size() == 1) // nothing to do
      { return; }
    
    Array<shared_ptr<BaseDOFMapStep>> sub_steps(steps.Size()),
      new_steps(steps.Size()), rss(0);
    new_steps.SetSize(0);
    for (auto k : Range(cutoffs.Size())) {
      int next = cutoffs[k];
      if (k == 0) {
	if (next != 0) throw Exception("FIRST CUTOFF MUST BE 0!!");
	continue;
      }
      int last = cutoffs[k-1];
      if (next == last+1) {
	new_steps.Append(steps[last]);
      }
      else { // next >= last+2
	sub_steps.SetSize(0);
	int curr = next-1;
	shared_ptr<BaseDOFMapStep> cstep = steps[curr];
	curr--; bool need_one = false;
	while (curr >= last) {
	  if ( auto cstep2 = steps[curr]->Concatenate(cstep) ) {
	    cstep = cstep2;
	    need_one = true;
	  }
	  else {
	    sub_steps.Append(cstep);
	    cstep = steps[curr];
	    need_one = true;
	  }
	  curr--;
	}
	if ( need_one ) {
	  sub_steps.Append(cstep);
	}
	if (sub_steps.Size()==1)
	  {
	    new_steps.Append(cstep);
	  }
	else {
	  rss.SetSize(sub_steps.Size());
	  for (auto l : Range(rss.Size()))
	    rss[l] = sub_steps[rss.Size()-1-l];
	  new_steps.Append(make_shared<ConcDMS>(rss));
	}
      }
    }
    steps = move(new_steps);
  }

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

  // template<class TMAT>
  // shared_ptr<BaseVector> ProlMap<TMAT>  :: CreateVector() const
  // {
  //   return make_shared<S_ParallelBaseVectorPtr<typename TMAT::TSCAL>>
  //     (pardofs->GetNDofLocal(), pardofs->GetEntrySize(), pardofs, DISTRIBUTED);
  // }

  // template<class TMAT>
  // shared_ptr<BaseVector> ProlMap<TMAT>  :: CreateMappedVector() const
  // {
  //   return make_shared<S_ParallelBaseVectorPtr<typename TMAT::TSCAL>>
  //     (mapped_pardofs->GetNDofLocal(), mapped_pardofs->GetEntrySize(), mapped_pardofs, DISTRIBUTED);
  // }


  template<class TMAT>
  void ProlMap<TMAT> :: TransferF2C (const shared_ptr<const BaseVector> & x_fine,
				     const shared_ptr<BaseVector> & x_coarse) const
  {
    RegionTimer rt(timer_hack_prol_f2c());
    // x_coarse->FVDouble() = 0.0;
    // prol->MultTransAdd(1.0, *x_fine, *x_coarse);
    x_fine->Distribute();
    prol_trans->Mult(*x_fine, *x_coarse);
    x_coarse->SetParallelStatus(DISTRIBUTED);
  }

  template<class TMAT>
  void ProlMap<TMAT> :: TransferC2F (const shared_ptr<BaseVector> & x_fine,
				     const shared_ptr<const BaseVector> & x_coarse) const
  {
    RegionTimer rt(timer_hack_prol_c2f());
    x_coarse->Cumulate();
    prol->Mult(*x_coarse, *x_fine);
    x_fine->SetParallelStatus(CUMULATED);
  }
  
  template<class TMAT>
  shared_ptr<BaseDOFMapStep> ProlMap<TMAT> :: Concatenate (shared_ptr<BaseDOFMapStep> other)
  {
    // cout << "conc me, " << this << " and " << other.get() << endl;
    // cout << "conc me, " << typeid(*this).name() << " with right " << typeid(*other).name() << endl;
    // me left, other right
    if (auto otmp = dynamic_pointer_cast<TwoProlMap<SPM_TM_C, SPM_TM_C>>(other)) {
      // cout << "conc, am dim " << GetProl()->Height() << " x " << GetProl()->Width()  << endl;
      // cout << "TPM, left is dim " << otmp->GetLMap()->GetProl()->Height() << " x " << otmp->GetLMap()->GetProl()->Width()  << endl;
      // cout << "TPM, right is dim " << otmp->GetRMap()->GetProl()->Height() << " x " << otmp->GetRMap()->GetProl()->Width()  << endl;
      // A - [B.., S(..)]
      auto comp_map = static_pointer_cast<ProlMap<mult_spm_tm<TMAT, SPM_TM_C>>>(Concatenate (otmp->GetLMap()));
      if (comp_map->IsPW()) { // [PP.., S(..)]
	auto tpm = make_shared<TwoProlMap<mult_spm_tm<TMAT, SPM_TM_C>, SPM_TM_C>> (comp_map, otmp->GetRMap());
	tpm->SetLog(LOGFUNC);
	return tpm;
      }
      else { // anything with S, S(...)
	auto comp_comp_prol = MatMultAB<mult_spm_tm<TMAT, SPM_TM_C>, SPM_TM_C> (*comp_map->GetProl(), *otmp->GetRMap()->GetProl());
	auto comp_comp_pmap = make_shared<ProlMap<mult_spm_tm<TMAT, SPM_TM_C>>> (comp_comp_prol, GetParDofs(), otmp->GetMappedParDofs(), false);
	comp_comp_pmap->SetCnt(comp_map->GetCnt() + GetCnt());
	comp_comp_pmap->SetLog(GetLog());
	return comp_comp_pmap;
      }
    }
    else if (auto opmap = dynamic_pointer_cast<ProlMap<SPM_TM_C>>(other)) {
      if (!has_lf) SetLog(opmap->GetLog()); // hack, why??
      // cout << "conc, am dim " << GetProl()->Height() << " x " << GetProl()->Width()  << endl;
      // cout << "other is dim " << opmap->GetProl()->Height() << " x " << opmap->GetProl()->Width()  << endl;
      if ( (!opmap->IsPW()) && opmap->CanSM()) {
	throw Exception("Right ProlMap should always be final!!");
	return nullptr;
      }

      if (opmap->IsPW() && opmap->CanSM() && !CanSM()) {
      	// this is a bit of a hack for the embed-step
      	// if I cannot smooth myself, and the right prol can, but has not, it should do that now
      	opmap->Smooth();
      	opmap->ClearSMF();
      }
      
      // (Sn) P - S(...)    -> return TPM [(Sn) P, S(..)]
      // P - S(...)    -> return TPM [P, S(..)]
      if ( (IsPW()) && (!MustSM()) && (!opmap->IsPW()) ) {
	auto tpm = make_shared<TwoProlMap<TMAT, SPM_TM_C>> (make_shared<ProlMap<TMAT>> (GetProl(), GetParDofs(), GetMappedParDofs()), opmap);
	tpm->GetLMap()->SetCnt(GetCnt());
	tpm->SetLog(LOGFUNC);
	return tpm;
      }
      // P - P         -> mult, return PMAP
      // (S)P - P      -> mult, smooth, return PMAP
      // (S)P - S(...) -> smooth, mult, returm PMAP
      // (Sn)P - ((Sn)) P -> mult, return PMAP with (Sn)
      // (Sn)p - ((Sn)) P -> mult, return PAMP wirh (Sn)
      bool smooth_now = IsPW() && MustSM();
      if ( smooth_now && (!opmap->IsPW()) )
	{ // (S)P - S(...)
	  smooth_now = false;
	  Smooth();
	}
      auto comp_prol = MatMultAB<TMAT, SPM_TM_C> (*prol, *opmap->GetProl());
      auto comp_map = make_shared<ProlMap<mult_spm_tm<TMAT, SPM_TM_C>>> (comp_prol, GetParDofs(), opmap->GetMappedParDofs(),
								   ( IsPW() && opmap->IsPW() ) );
      comp_map->SetCnt(GetCnt() + opmap->GetCnt());
      comp_map->SetLog(LOGFUNC);
      if ( smooth_now )
	{ // (S)P - P
	  comp_map->SetSMF(GetSMF());
	  comp_map->Smooth();
	  ClearSMF();
	}
      else if (CanSM()) {
	comp_map->SetSMF(GetSMF(), MustSM());
	ClearSMF();
      }
      // cout << "resulting prol sm, can, must" << !comp_map->IsPW() << " " << comp_map->CanSM() << " " << comp_map->MustSM() << endl;
      // cout << "resulting prol sm? " << !comp_map->IsPW() << endl;
      return comp_map;
    }
    else {
      return nullptr;
    }
  }

  template<class TMAT>
  void ProlMap<TMAT> :: SetSMF ( std::function<void(ProlMap<TMAT> * map)> ASMFUNC, bool forced)
  {
    if (!IsPW()) { throw Exception("Cannot set smoothed (Already smoothed)."); }
    // cout << "set smoothed, am dim " << GetProl()->Height() << " x " << GetProl()->Width()  << endl;
    SMFUNC = ASMFUNC; has_smf = true; force_sm = forced;
  }

  template<class TMAT>
  void ProlMap<TMAT> :: ClearSMF ( )
  {
    // cout << "clear smoothed, am dim " << GetProl()->Height() << " x " << GetProl()->Width()  << endl;
    SMFUNC = [](auto x){ ; }; has_smf = false; force_sm = false;
  }

  template<class TMAT>
  void ProlMap<TMAT> :: SetLog ( std::function<void(shared_ptr<BaseSparseMatrix> prol)> ALOGFUNC)
  {
    LOGFUNC = ALOGFUNC; has_lf = true;
  }

  template<class TMAT>
  void ProlMap<TMAT> :: ClearLog ()
  {
    LOGFUNC = [&](auto x){ ; }; has_lf = true;
  }
  
  template<class TMAT>
  void ProlMap<TMAT> :: Smooth ()
  {
    if (!IsPW()) { throw Exception("Cannot smooth twice."); }
    // if (!WantSM()) { throw Exception("Cannot smooth (not set)."); }
    // cout << "call smooth, am dim " << GetProl()->Height() << " x " << GetProl()->Width()  << ", cnt is " << GetCnt() << endl;
    // cout << "call smooted with cnt " << GetCnt() << endl;
    //amg->SmoothProlongation_hack (this, mesh);
    SMFUNC(this); ispw = false;
    ClearSMF();
  }
  
  template<class TMAT>
  shared_ptr<BaseSparseMatrix> ProlMap<TMAT> :: AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const
  {
    // cout << "assemble with cnt " << GetCnt() << endl;
    if (IsPW() && CanSM() ) {
      // cout << "Call smooth before assmat!" << endl;
      const_cast<ProlMap<TMAT>&>(*this).Smooth();
    }
    auto tfmat = dynamic_pointer_cast<SPM_TM_F>(mat);
    if (tfmat==nullptr) {
      throw Exception(string("Cannot cast to ") + typeid(SPM_TM_F).name() + string(", type = ") + typeid(*mat).name() + string("! !") );
      return nullptr;
    }

    if (has_lf)
      { /* cout << "call log-func" << endl; */ LOGFUNC(GetProl()); }
    else
      { throw Exception("NO LOGFUNC!!");}
    auto& self = const_cast<ProlMap<TMAT>&>(*this);
    self.prol_trans = TransposeSPM(*prol);
    self.prol = make_shared<SPM_P>(move(*prol));
    self.prol_trans = make_shared<trans_spm<SPM_P>>(move(*prol_trans));
    

    // cout << prol->Height() << " x " << prol->Width() << endl;
							// cout << tfmat->Height() << " x " << tfmat->Width() << endl;


							// 						      print_tm_spmat(cout, *prol); cout << endl;
    
    auto spm_tm = RestrictMatrixTM<SPM_TM_F, TMAT> (*prol_trans, *tfmat, *prol);
    return make_shared<SPM_C>(move(*spm_tm));
  }

} // namespace amg

#include "amg_tcs.hpp"
