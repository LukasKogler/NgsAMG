#define FILE_AMGMAP_CPP

#include "amg.hpp"

namespace amg
{

  void DOFMap :: SetCutoffs (FlatArray<size_t> acutoffs)
  {
    cutoffs.SetSize(acutoffs.Size());
    cutoffs = acutoffs;
    ConcSteps();
  }
    
  void DOFMap :: ConcSteps ()
  {
    // cout << "CONCSTEPS: " << endl; prow2(steps); cout << endl;
    static Timer t("DOFMap::ConcSteps");
    RegionTimer rt(t);
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
    x_coarse->FVDouble() = 0.0;
    prol->MultTransAdd(1.0, *x_fine, *x_coarse);
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
    if (auto opmap = dynamic_pointer_cast<ProlMap<TCMAT>>(other)) {
      if ( (!opmap->IsPW()) && opmap->WantSM()) {
	throw Exception("Right ProlMap should always be final!!");
	return nullptr;
      }
      // P - S(...)    -> return TPM [P, S(..)]
      if ( (IsPW()) && (!WantSM()) && (!opmap->IsPW()) ) {
	return make_shared<TwoProlMap<TMAT, TCMAT>> (make_shared<ProlMap<TMAT>> (GetProl(), GetParDofs(), GetMappedParDofs()), opmap);
      }
      // P - P         -> mult, return PMAP
      // (S)P - P      -> mult, smooth, return PMAP
      // (S)P - S(...) -> smooth, mult, returm PMAP
      if ( IsPW() && WantSM() && (!opmap->IsPW()) )
	{ // (S)P - S(...)
	  Smooth();
	}
      auto comp_prol = MatMultAB<TMAT, TCMAT> (*prol, *opmap->GetProl());
      auto comp_map = make_shared<ProlMap<mult_spm<TMAT, TCMAT>>> (comp_prol, GetParDofs(), opmap->GetMappedParDofs(),
								   (IsPW() && opmap->IsPW()) );
      comp_map->SetSmoothed(SMFUNC);
      if ( comp_map->WantSM() )
	{ // (S)P - P
	  comp_map->Smooth();
	}
      return comp_map;
    }
    else if (auto otmp = dynamic_pointer_cast<TwoProlMap<TCMAT, TCMAT>>(other)) {
      auto comp_map = static_pointer_cast<ProlMap<mult_spm<TMAT, TCMAT>>>(Concatenate (otmp->GetLMap()));
      // P - [P.., S(..)]
      if (comp_map->IsPW()) { // [PP.., S(..)]
	return make_shared<TwoProlMap<mult_spm<TMAT, TCMAT>, TCMAT>> (comp_map, otmp->GetRMap());
      }
      else { // S(PP..), S(...)
	auto comp_comp_prol = MatMultAB<mult_spm<TMAT, TCMAT>, TCMAT> (*comp_map->GetProl(), *otmp->GetRMap()->GetProl());
	return make_shared<ProlMap<mult_spm<TMAT, TCMAT>>> (comp_comp_prol, GetParDofs(), otmp->GetMappedParDofs(), false);
      }
    }
    else {
      return nullptr;
    }
  }

  
  // template<class TMAT>
  // void ProlMap<TMAT> :: SetSmoothed (const VWiseAMG* aamg, shared_ptr<TopologicMesh> amesh)
  // {
  //   if (!IsPW()) { throw Exception("Cannot set smoothed (Already smoothed)."); }
  //   amg = aamg; mesh = amgesh;
  // }

  template<class TMAT>
  void ProlMap<TMAT> :: SetSmoothed ( std::function<void(ProlMap<TMAT> * map)> ASMFUNC)
  {
    if (!IsPW()) { throw Exception("Cannot set smoothed (Already smoothed)."); }
    SMFUNC = ASMFUNC; has_smf = true; // mesh = amesh;
  }

  
  template<class TMAT>
  void ProlMap<TMAT> :: Smooth ()
  {
    if (!IsPW()) { throw Exception("Cannot smooth twice."); }
    if (!WantSM()) { throw Exception("Cannot smooth (not set)."); }
    //amg->SmoothProlongation_hack (this, mesh);
    SMFUNC(this);
    SMFUNC = [](auto x) { ; };
    has_smf = false;
    ispw = false;
  }
  
  template<class TMAT>
  shared_ptr<BaseSparseMatrix> ProlMap<TMAT> :: AssembleMatrix (shared_ptr<BaseSparseMatrix> mat) const
  {
    auto tfmat = dynamic_pointer_cast<TFMAT>(mat);
    if (tfmat==nullptr) {
      throw Exception(string("Cannot cast to ") + typeid(TFMAT).name() + string("!!") );
      return nullptr;
    }
    return RestrictMatrix<TFMAT, TMAT> (*tfmat, *prol);
  }

} // namespace amg

#include "amg_tcs.hpp"
