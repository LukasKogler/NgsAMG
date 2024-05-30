
#ifdef USE_TAU
#include "TAU.h"
#endif

#include <base.hpp>
#include <utils_io.hpp>
#include <utils_sparseLA.hpp>
#include <utils_sparseMM.hpp>

#include "dof_map.hpp"

namespace amg
{

shared_ptr<BaseDOFMapStep>
MakeSingleStep2 (FlatArray<shared_ptr<BaseDOFMapStep>> init_steps)
{
  const int iss = init_steps.Size();
  if (iss == 0)
    { return nullptr; }
  Array<shared_ptr<BaseDOFMapStep>> sub_steps(iss); sub_steps.SetSize0();
  for (int k = 0; k < iss; k++) {
    shared_ptr<BaseDOFMapStep> conc_step = init_steps[k];
    int j = k+1;
    for ( ; j < iss; j++) {
      if (auto x = conc_step->Concatenate(init_steps[j]))
        { conc_step = x; k++; }
      else
        { break; }
    }
    sub_steps.Append(conc_step);
  }
  shared_ptr<BaseDOFMapStep> final_step = nullptr;
  if (sub_steps.Size() == 1)
    { return sub_steps[0]; }
  else
    { return make_shared<ConcDMS>(sub_steps); }
} // MakeSingleStep2


Array<shared_ptr<BaseDOFMapStep>>
GetSimpleSteps(shared_ptr<BaseDOFMapStep> step)
{
  Array<shared_ptr<BaseDOFMapStep>> simpleSteps;
  if (auto concDMS = dynamic_pointer_cast<ConcDMS>(step))
  {
    int nSteps = concDMS->GetNSteps();
    for (auto j : Range(nSteps))
    {
      simpleSteps.Append(GetSimpleSteps(concDMS->GetStep(j)));
    }
  }
  else if (auto multiStep = dynamic_pointer_cast<MultiDofMapStep>(step))
  {
    simpleSteps.Append(GetSimpleSteps(multiStep->GetMap(0)));
  }
  else
  {
    simpleSteps.Append(step);
  }
  return simpleSteps;
} // GetSimpleSteps


shared_ptr<BaseDOFMapStep>
MakeSingleStep3 (FlatArray<shared_ptr<BaseDOFMapStep>> init_steps)
{
  Array<shared_ptr<BaseDOFMapStep>> simpleSteps;

  for (auto k : Range(init_steps))
  {
    simpleSteps.Append(GetSimpleSteps(init_steps[k]));
  }

  return MakeSingleStep2(simpleSteps);
} // MakeSingleStep3


/** DOFMap **/

DOFMap::
DOFMap ()
  : finalized(false)
  , nsteps_glob(-1)
{}

void
DOFMap::
Finalize ()
{
  if ( (steps.Size() == 0) || finalized )
    { return; }
  int nsteps_loc = steps.Size();
  if (GetUDofs().IsParallel())
    { nsteps_glob = GetUDofs().GetCommunicator().AllReduce(nsteps_loc, NG_MPI_MAX); }
  vecs.SetSize(nsteps_loc + 1);
  for (auto k : Range(nsteps_loc))
    { vecs[k] = GetStep(k)->CreateVector(); }
  if ( nsteps_glob == nsteps_loc ) // can be nullptr!
    { vecs.Last() = steps.Last()->CreateMappedVector(); }
  finalized = true;
  // EMPTY glob dnums
  have_dnums.SetSize(GetNLevels()); have_dnums.Clear();
  glob_dnums.SetSize(GetNLevels());
} // DOFMap::Finalize

void DOFMap :: AddStep (const shared_ptr<BaseDOFMapStep> step)
{
  if (finalized)
    throw Exception("Tried to add a step to a finalized DofMap!!");
  steps.Append(step);
} // DOFMap :: AddStep


void
DOFMap::
ReplaceStep (int k, shared_ptr<BaseDOFMapStep> newStep)
{
  /**
   *  This is a hack for HDIV AMG. During smoother setup, the matrices are
   *  replaced by dyn-block matrices, and the prol-maps by dyn-block prol-maps.
   *
   *  The DOF-blocking of an AMG-level is the one for the pre-contracted
   *  level, so the dof-blocking for the contracted level is only available
   *  AFTER the matrix has been converted. Therefore, we can only replace
   *  the prol-maps at that point and not before.
   *
   *  It is not clean AT ALL, but supporting dyn-block matrices fully would require:
   *      -) mat-mat multiplication dyn-block x dyn-block (implemented, not tested)
   *      -) mat-mat multiplication dyn-block x sparse 
   *      -) contract dyn-block matrix
   *      -) dyn-block -> hybrid dyn-block conversion
   *      -) create SparseMatrix<double> pot-mat from curl-mat and dyn-sparse range-mat
   *      -) ideally, directly assemble dyn-sparse prol
   *      -) ideally, directly assemble dyn-sparse emb-prol
   *      -) all the routing for this stuff...
   *      -) dyn-block inverse
   *
   *  If it was properly supported, that would ofc not only be cleaner but also speed
   *  up the setup a lot, but it is not feasible to implement everythign right now.
   */
  steps[k] = newStep;
} // DOFMap::ReplaceStep


Array<shared_ptr<BaseMatrix>>
DOFMap::
AssembleMatrices (shared_ptr<BaseMatrix> finest_mat) const
{
  static Timer t("DOFMap::AssembleMatrices");
  RegionTimer rt(t);
  const_cast<DOFMap &>(*this).Finalize();
  shared_ptr<BaseMatrix> last_mat = finest_mat;
  Array<shared_ptr<BaseMatrix>> mats;
  mats.Append(last_mat);
  // cout << "SS IS : " << steps.Size() << endl;
  for (auto step_nr : Range(steps.Size())) {
    auto & step = steps[step_nr];
    auto next_mat = step->AssembleMatrix(last_mat);
    mats.Append(next_mat);
    last_mat = next_mat;
  }
  return mats;
} // DOFMap::AssembleMatrices


void
DOFMap::
TransferAtoB (int la, int lb, const BaseVector * vin, BaseVector * vout) const
{
  // cout << " go " << la << " <-> " << lb << endl;
  if (la == lb)
    { *vout = *vin; return; }
  int maxlev_loc = steps.Size();;
  int lla = min(la, maxlev_loc), llb = min(lb, maxlev_loc);
  if (lla == llb) // transfer between levels I dont have
    { return; }
  bool c2f = (la > lb);
  if (c2f)
    { swap(lla, llb); }
  // cout << " becomes " << lla << " <-> " << llb << endl;
  auto submap = const_cast<DOFMap&>(*this).ConcStep(lla, llb);
  if (c2f)
    { submap->TransferC2F(vout, vin); }
  else
    { submap->TransferF2C(vin, vout); }
} // DOFMap::TransferAtoB


shared_ptr<DOFMap>
DOFMap::
SubMap (int from, int to) const
{
  auto submap = make_shared<DOFMap>();
  int kmax = (to == -1) ? steps.Size() : to;
  for (auto k : Range(from, kmax))
    { submap->AddStep(steps[k]); }
  return submap;
} // DOFMap::SubMap


shared_ptr<BaseDOFMapStep>
DOFMap::
ConcStep (int la, int lb, bool symbolic) const
{
  const_cast<DOFMap&>(*this).Finalize();
  // cout << " concstep " << la << " " << lb << " of " << steps.Size() << ", symbolic = " << symbolic << endl;
  if (la > lb)
    { swap(la, lb); }
  if (la+1 == lb)
    { return steps[la]; }
  // cout << " R " << la << " " << lb << ", vecs " << la+1 << " " << lb-1 << endl;
  if (symbolic)
  {
    return make_shared<ConcDMS>(steps.Range(la, lb), vecs.Range(la+1, lb));
  }
  else
  {
    // Array<shared_ptr<BaseDOFMapStep>> simpleSteps;
    // for (auto k : Range(la, lb))
    // {
    //   if (auto mdStep = dynamic_pointer_cast<MultiDofMapStep>(steps[k]))
    //   {
    //     cout << " step " << k << " is multi " << endl;
    //     auto simpleStep = mdStep->GetMap(0);
    //     simpleSteps.Append(simpleStep);
    //     cout << "   simple-step type = " << typeid(*(mdStep->GetMap(0))).name() << endl;
    //   }
    //   else
    //   {
    //     cout << " step " << k << " is NOT multi " << endl;
    //     simpleSteps.Append(steps[k]);
    //   }
    // }
    return MakeSingleStep3(steps.Range(la, lb));
  }
  // return make_shared<ConcDMS>(steps.Range(la, lb-1), vecs.Range(la+1, lb-1));
} // DOFMap::ConcStep

FlatArray<int>
DOFMap::
GetGlobDNums (int level) const
{
  const_cast<DOFMap&>(*this).Finalize();

  if (!have_dnums.Test(level))
  {
    auto pd = GetParDofs(level);
    auto all = make_shared<BitArray>(pd->GetNDofLocal()); all->Set();
    int gn;
    pd->EnumerateGlobally(all, glob_dnums[level], gn);
    const_cast<BitArray&>(have_dnums).SetBit(level);
    // cout << " GLOB ENUM LEVEL " << level << endl;
    // prow2(glob_dnums); cout << endl;
  }

  return glob_dnums[level];
} // DOFMap::GetGlobDNums


std::ostream & operator<<(std::ostream &os, const BaseDOFMapStep& p)
{
  p.PrintTo(os);
  return os;
}

/** END DofMap **/


/** BaseDOFMapStep **/

BaseDOFMapStep::
BaseDOFMapStep(UniversalDofs originDofs,
               UniversalDofs mappedDofs)
  : _originDofs(originDofs)
  , _mappedDofs(mappedDofs)
{}


unique_ptr<BaseVector>
BaseDOFMapStep::
CreateVector () const
{
  return _originDofs.CreateVector();
} // BaseDOFMapStep::CreateVector


unique_ptr<BaseVector>
BaseDOFMapStep::
CreateMappedVector () const
{
  return _mappedDofs.CreateVector();
} // BaseDOFMapStep::CreateMappedVector

/** END BaseDOFMapStep **/


/** ConcDMS **/

ConcDMS::
ConcDMS (FlatArray<shared_ptr<BaseDOFMapStep>> _sub_steps,
         FlatArray<shared_ptr<BaseVector>> _vecs)
  : BaseDOFMapStep(_sub_steps[0]->GetUDofs(), _sub_steps.Last()->GetMappedUDofs())
  , sub_steps(_sub_steps.Size())
  , spvecs(0)
  , vecs(0)
{
  sub_steps = _sub_steps;

  if( ( sub_steps.Size() > 1 ) &&
      ( _vecs.Size() != 0 ) )
  {
    spvecs.SetSize(sub_steps.Size()-1);

    for(auto k : Range(spvecs))
      { spvecs[k] = _vecs[k]; }

    vecs.SetSize(spvecs.Size());

    for (auto k : Range(spvecs.Size()))
      { vecs[k] = spvecs[k].get(); }
  }
  else
  {
    vecs = nullptr;
  }
} // ConcDMS(..)


void
ConcDMS::
Finalize ()
{
  for (auto step : sub_steps)
    { step->Finalize(); }

  if ( ( sub_steps.Size() > 1 ) &&
       ( spvecs.Size() == 0 ) )
  {
    spvecs.SetSize(sub_steps.Size()-1);

    for(auto k : Range(spvecs))
      { spvecs[k] = sub_steps[k]->CreateMappedVector(); }

    vecs.SetSize(spvecs.Size());

    for (auto k : Range(spvecs.Size()))
      { vecs[k] = spvecs[k].get(); }
  }

} // ConcDMS::Finalize


void
ConcDMS::
TransferF2C (BaseVector const *x_fine, BaseVector *x_coarse) const
{
#ifdef USE_TAU
  TAU_PROFILE("ConcDMS::TransferF2C", TAU_CT(*this), TAU_DEFAULT);
#endif

  if (sub_steps.Size() == 1)
    { sub_steps[0]->TransferF2C(x_fine, x_coarse); }
  else {
    sub_steps[0]->TransferF2C(x_fine, vecs[0]);
    for (int l = 1; l<int(sub_steps.Size())-1; l++)
      { sub_steps[l]->TransferF2C(vecs[l-1], vecs[l]); }
    sub_steps.Last()->TransferF2C(vecs.Last(), x_coarse);
  }
} // ConcDMS::TransferF2C


void
ConcDMS::
AddF2C (double fac, BaseVector const *x_fine, BaseVector *x_coarse) const
{
#ifdef USE_TAU
  TAU_PROFILE("ConcDMS::AddF2C", TAU_CT(*this), TAU_DEFAULT);
#endif
  if (sub_steps.Size() == 1)
    { sub_steps[0]->AddF2C(fac, x_fine, x_coarse); }
  else {
    sub_steps[0]->TransferF2C(x_fine, vecs[0]);
    for (int l = 1; l<int(sub_steps.Size())-1; l++)
      { sub_steps[l]->TransferF2C(vecs[l-1], vecs[l]); }
    sub_steps.Last()->AddF2C(fac, vecs.Last(), x_coarse);
  }
} // ConcDMS::AddF2C


void
ConcDMS::
TransferC2F (BaseVector *x_fine, BaseVector const *x_coarse) const
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
      { sub_steps[l]->TransferC2F(vecs[l-1], vecs[l]); }
    sub_steps[0]->TransferC2F(x_fine, vecs[0]);
  }
} // ConcDMS::TransferC2F


void
ConcDMS::
AddC2F (double fac, BaseVector *x_fine, BaseVector const *x_coarse) const
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
      { sub_steps[l]->TransferC2F(vecs[l-1], vecs[l]); }
    sub_steps[0]->AddC2F(fac, x_fine, vecs[0]);
  }
} // ConcDMS::AddC2F


shared_ptr<BaseMatrix>
ConcDMS::
AssembleMatrix (shared_ptr<BaseMatrix> mat) const
{
  shared_ptr<BaseMatrix> cmat = mat;
  for (auto& step : sub_steps)
    { cmat = step->AssembleMatrix(cmat); }
  return cmat;
}


void
ConcDMS::
PrintTo (std::ostream & os, string prefix) const
{
  os << prefix << " ConcDMS with " << sub_steps.Size() << " steps " << endl;
  for (auto k : Range(sub_steps)) {
    os << prefix << " step # " << k << ": " << endl;
    sub_steps[k]->PrintTo(os, prefix + "  ");
    os << endl;
  }
}


int
ConcDMS::
GetBlockSize () const
{
  return sub_steps[0]->GetBlockSize();
} // ConcDMS::GetBlockSize


int
ConcDMS::
GetMappedBlockSize () const
{
  return sub_steps.Last()->GetMappedBlockSize();
} // ConcDMS::GetMappedBlockSize

/** END ConcDMS **/


/** BaseProlMap **/

void
BaseProlMap::
TransferF2C (BaseVector const *x_fine,
             BaseVector       *x_coarse) const
{
  x_fine->Distribute();
  x_coarse->SetParallelStatus(DISTRIBUTED);

  if (auto prolT = GetBaseProlTrans())
  {
    prolT->Mult(*x_fine->GetLocalVector(),
                *x_coarse->GetLocalVector());
  }
  else
  {
    GetBaseProl()->MultTrans(*x_fine->GetLocalVector(),
                             *x_coarse->GetLocalVector());

  }
}


void
BaseProlMap::
AddF2C (double fac, BaseVector const *x_fine, BaseVector *x_coarse) const
{
  x_fine->Distribute();
  x_coarse->Distribute();

  if (auto prolT = GetBaseProlTrans())
  {
    prolT->MultAdd(fac,
                   *x_fine->GetLocalVector(),
                   *x_coarse->GetLocalVector());
  }
  else
  {
    GetBaseProl()->MultTransAdd(fac,
                                *x_fine->GetLocalVector(),
                                *x_coarse->GetLocalVector());
  }
}


void
BaseProlMap::
TransferC2F (BaseVector       *x_fine,
             BaseVector const *x_coarse) const
{
  x_coarse->Cumulate();
  x_fine->SetParallelStatus(CUMULATED);

  GetBaseProl()->Mult(*x_coarse->GetLocalVector(),
                      *x_fine->GetLocalVector());
} // BaseProlMap::TransferC2F


void
BaseProlMap::
AddC2F (double           fac,
        BaseVector       *x_fine,
        BaseVector const *x_coarse) const
{
  x_coarse->Cumulate();
  x_fine->Cumulate();

  GetBaseProl()->MultAdd(fac,
                         *x_coarse->GetLocalVector(),
                         *x_fine->GetLocalVector());
} // BaseProlMap::AddC2F

/** END BaseProlMap **/


/** ProlMap **/

template<class TM>
ProlMap<TM>::
ProlMap (shared_ptr<TMAT> prol,
         shared_ptr<typename ProlMap<TM>::SPM_R> prolT,
         UniversalDofs originDofs,
         UniversalDofs mappedDofs)
  : BaseSparseProlMap(originDofs, mappedDofs)
  , _prol(prol)
  , _prolT(prolT)
{}


template<class TM>
ProlMap<TM>::
ProlMap (shared_ptr<TMAT> aprol,
         UniversalDofs originDofs,
         UniversalDofs mappedDofs)
  : ProlMap<TM>(aprol, nullptr, originDofs, mappedDofs)
{}


template<class TM>
void
ProlMap<TM>::
Finalize ()
{
  this->BuildPT();

  if (dynamic_pointer_cast<SPM_P>(_prol) == nullptr)
    { _prol = make_shared<SPM_P>(std::move(*_prol)); }

  if (dynamic_pointer_cast<SPM_R>(_prolT) == nullptr)
    { _prolT = make_shared<SPM_R>(std::move(*_prolT)); }
} // ProlMap::Finalize


INLINE Timer<>&
timer_hack_prol_f2c ()
{
  static Timer t("ProlMap::TransferF2C");

  return t;
} // timer_hack_prol_f2c

INLINE Timer<>&
timer_hack_prol_c2f ()
{
  static Timer t("ProlMap::TransferC2F");

  return t;
} // timer_hack_prol_c2f


template<class TM>
void
ProlMap<TM>::
TransferF2C (BaseVector const *x_fine,
             BaseVector *x_coarse) const
{
#ifdef USE_TAU
  TAU_PROFILE("ProlMap::TransferF2C", TAU_CT(*this), TAU_DEFAULT);
#endif

  RegionTimer rt(timer_hack_prol_f2c());

  x_fine->Distribute();

  _prolT->Mult(*x_fine, *x_coarse);

  x_coarse->SetParallelStatus(DISTRIBUTED);
} // ProlMap::TransferF2C


template<class TM>
void
ProlMap<TM>::
AddF2C (double fac, BaseVector const *x_fine, BaseVector *x_coarse) const
{
#ifdef USE_TAU
  TAU_PROFILE("ProlMap::AddF2C", TAU_CT(*this), TAU_DEFAULT);
#endif

  RegionTimer rt(timer_hack_prol_f2c());

  x_fine->Distribute();
  x_coarse->Distribute();

  _prolT->MultAdd(fac, *x_fine, *x_coarse);
} // ProlMap::AddF2C


template<class TM>
void
ProlMap<TM>::
TransferC2F (BaseVector *x_fine, BaseVector const *x_coarse) const
{
#ifdef USE_TAU
  TAU_PROFILE("ProlMap::TransferC2F", TAU_CT(*this), TAU_DEFAULT);
#endif

  RegionTimer rt(timer_hack_prol_c2f());

  x_coarse->Cumulate();

  _prol->Mult(*x_coarse, *x_fine);

  x_fine->SetParallelStatus(CUMULATED);
} // ProlMap::TransferC2F


template<class TM>
void
ProlMap<TM>::
AddC2F (double fac, BaseVector *x_fine, BaseVector const *x_coarse) const
{
#ifdef USE_TAU
  TAU_PROFILE("ProlMap::AddC2F", TAU_CT(*this), TAU_DEFAULT);
#endif

  RegionTimer rt(timer_hack_prol_c2f());

  x_coarse->Cumulate();
  x_fine->Cumulate();

  _prol->MultAdd(fac, *x_coarse, *x_fine);
} // ProlMap::AddC2F


template<class TM>
shared_ptr<BaseDOFMapStep>
ProlMap<TM>::
ConcatenateFromLeft (BaseSparseProlMap &other)
{
  if (auto opmap = dynamic_cast<ProlMap<TM_F>*>(&other))
  {
    auto comp_prol = MatMultAB(*opmap->GetProl(), *_prol);

    auto comp_map = make_shared<ProlMap<TM>> (comp_prol, opmap->GetUDofs(), GetMappedUDofs());

    comp_map->Finalize();

    return comp_map;
  }
  else
  {
    return nullptr;
  }
} // ProlMap::ConcatenateFromLeft


template<class TM>
shared_ptr<BaseDOFMapStep>
ProlMap<TM>::
Concatenate (shared_ptr<BaseDOFMapStep> other)
{
  shared_ptr<BaseDOFMapStep> concStep = nullptr;

  if (auto otherPM = dynamic_pointer_cast<BaseSparseProlMap>(other))
  {
    otherPM->Finalize(); // make suare we have SPM, not only SPMTM

    DispatchOverMatrixDimensions(*otherPM->GetBaseProl(), [&](auto const &spO, auto OH, auto OW)
    {
      // if constexpr( (W == OH) &&
      //               isSparseMatrixCompiled<H, OW>() &&
      //               IsSparseMMCompiled<H, W, OW>() )
      if constexpr( (W == OH) &&
                    (OH == OW) &&
                    isSparseMatrixCompiled<H, OW>() )
      {
        auto concP = MatMultAB(*GetProl(), spO);

        concStep = make_shared<ProlMap<StripTM<H, OW>>>(concP,
                                                        GetUDofs(),
                                                        other->GetMappedUDofs());
      }
    });

    if (concStep == nullptr)
    {
      concStep = otherPM->ConcatenateFromLeft(*this);
    }
  }

  return concStep;
} // ProlMap::Concatenate


template<class TM>
shared_ptr<typename ProlMap<TM>::SPM_P>
ProlMap<TM>::
GetProl () const
{
  return _prol;
} // ProlMap::GetProl


template<class TM>
shared_ptr<typename ProlMap<TM>::SPM_R>
ProlMap<TM>::
GetProlTrans () const
{
  const_cast<ProlMap<TM>&>(*this).BuildPT();
  return _prolT;
} // ProlMap::GetProlTrans


template<class TM>
void
ProlMap<TM>::
SetProl (shared_ptr<TMAT> prol)
{
  if (_prol != prol) {
    _prol = prol;
    _prolT = nullptr;
  }
} // ProlMap::SetProl


template<class TM>
void
ProlMap<TM>::
BuildPT (bool force)
{
  if ( force || ( _prolT == nullptr ) )
    { _prolT = TransposeSPM(*_prol); }
} // ProlMap::BuildPT


template<class TMAT>
shared_ptr<BaseMatrix>
ProlMap<TMAT>::
AssembleMatrix (shared_ptr<BaseMatrix> mat) const
{
  // cout << "prolmap assmat, type " << typeid(*this).name() << endl;
  // cout << "prol type " << typeid(*_prol).name() << endl;
  // cout << "prol dims " << _prol->Height() << " x " << _prol->Width() << endl;
  // cout << "prol, fmat " << _prol << ", " << mat << endl;
  // cout << "fmat type " << typeid(*mat).name() << endl;
  // cout << "fmat dims " << mat->Height() << " x " << mat->Width() << endl;

  auto tfmat = my_dynamic_pointer_cast<SPM_TM_F>(mat, "ProlMap::AssembleMatrix - SPM");

  const_cast<ProlMap<TM>&>(*this).Finalize();;

  return RestrictMatrix<H, W> (*_prolT, *tfmat, *_prol);
} // ProlMap::AssembleMatrix


template<class TM>
void
ProlMap<TM>::
PrintTo (std::ostream & os, string prefix) const
{
  os << prefix << " ProlMap, blocks = (" << Height<typename TMAT::TENTRY>() << " x " <<
    Width<typename TMAT::TENTRY>() << "), prol dims = " << GetProl()->Height() << " x " << GetProl()->Width() << endl;
  os << prefix << " prol : " << endl;
  print_tm_spmat(os, *GetProl());
  os << endl;
  os << prefix << " trans prol : " << endl;
  print_tm_spmat(os, *GetProlTrans());
  os << endl;
} // ProlMap::PrintTo


template<class TMAT>
int ProlMap<TMAT>::
GetBlockSize () const
{
  return Height<typename TMAT::TENTRY>();
} // ProlMap::GetBlockSize


template<class TMAT>
int ProlMap<TMAT>::
GetMappedBlockSize () const
{
  return Width<typename TMAT::TENTRY>();
} // ProlMap::GetMappedBlockSize

/** END ProlMap **/


/** DynBlockProlMap **/

template<class TSCAL>
DynBlockProlMap<TSCAL>::
DynBlockProlMap (shared_ptr<DynBlockSparseMatrix<TSCAL>> prol,
                 UniversalDofs originDofs,
                 UniversalDofs mappedDofs,
                 bool buildPT)
  : BaseProlMap(originDofs, mappedDofs)
  , _prol(prol)
  , _prolT(nullptr)
{
  if (buildPT)
  {
    _prolT = nullptr;
  }
} // DynBlockProlMap(..)


template<class TSCAL>
DynBlockProlMap<TSCAL>::
DynBlockProlMap (shared_ptr<DynBlockSparseMatrix<TSCAL>> prol,
                  shared_ptr<DynBlockSparseMatrix<TSCAL>> prolT,
                  UniversalDofs originDofs,
                  UniversalDofs mappedDofs)
  : BaseProlMap(originDofs, mappedDofs)
  , _prol(prol)
  , _prolT(prolT)
{}


template<class TSCAL>
bool
DynBlockProlMap<TSCAL>::
CanConcatenate (shared_ptr<BaseDOFMapStep> other)
{
  return false;
} // DynBlockProlMap::CanConcatenate


template<class TSCAL>
shared_ptr<BaseDOFMapStep>
DynBlockProlMap<TSCAL>::
Concatenate (shared_ptr<BaseDOFMapStep> other)
{
  return nullptr;
} // DynBlockProlMap::Concatenate


template<class TSCAL>
shared_ptr<BaseDOFMapStep>
DynBlockProlMap<TSCAL>::
PullBack (shared_ptr<BaseDOFMapStep> other)
{
  /** "me - other" -> "new other - new me" **/
  return nullptr;
} // DynBlockProlMap::PullBack


template<class TSCAL>
shared_ptr<BaseMatrix>
DynBlockProlMap<TSCAL>::
AssembleMatrix (shared_ptr<BaseMatrix> mat) const
{
  throw Exception("Called DynBlockProlMap::AssembleMatrix - not implemented!");
} // DynBlockProlMap::AssembleMatrix

  
template<class TSCAL>
void
DynBlockProlMap<TSCAL>::
Finalize ()
{
  ;
}


template<class TSCAL>
void
DynBlockProlMap<TSCAL>::
PrintTo (std::ostream & os, string prefix ) const
{
  os << prefix << "DynBlockProlMap " << endl;
  os << prefix << "  _prol: " << _prol << endl;
  os << prefix << "  _prolT: " << _prolT << endl;
  os << prefix << "  Prolongation: " << *_prol << endl;

  if ( _prolT )
  {
    os << prefix << "  Transposed Prolongation: " << _prolT << endl;
  }
  else
  {
    os << prefix << "  Transposed Prolongation not stored seperately!" << endl;
  }
} // DynBlockProlMap::PrintTo


template<class TSCAL>
int
DynBlockProlMap<TSCAL>::
GetBlockSize ()       const
{
  return 1;
} // DynBlockProlMap::GetBlockSize


template<class TSCAL>
int
DynBlockProlMap<TSCAL>::
GetMappedBlockSize () const
{
  return 1;
} // DynBlockProlMap::GetMappedBlockSize

/** END DynBlockProlMap **/


shared_ptr<DynBlockProlMap<double>>
ConvertToDynSparseProlMap(ProlMap<double>     const &sparseProlMap,
                          DynVectorBlocking<> const &fineBlocking,
                          DynVectorBlocking<> const &coarseBlocking)
{
  auto dynBlockProl = make_shared<DynBlockSparseMatrix<double>>(*sparseProlMap.GetProl(),
                                                                fineBlocking,
                                                                coarseBlocking,
                                                                false);

  return make_shared<DynBlockProlMap<double>>(dynBlockProl,
                                              sparseProlMap.GetUDofs(),
                                              sparseProlMap.GetMappedUDofs(),
                                              false);
} // ConvertToDynSparseProlMap


/** MultiDofMapStep **/

MultiDofMapStep::
MultiDofMapStep (FlatArray<shared_ptr<BaseDOFMapStep>> _maps)
  : BaseDOFMapStep(_maps[0]->GetUDofs(), _maps[0]->GetMappedUDofs())
{
  maps.SetSize(_maps.Size());
  maps = _maps;
} // MultiDofMapStep(..)


void
MultiDofMapStep::
Finalize ()
{
  for (auto map : maps)
    { map->Finalize(); }
} // MultiDofMapStep::Finalize


void
MultiDofMapStep::
TransferF2C (BaseVector const *x_fine, BaseVector *x_coarse) const
{
  GetPrimMap()->TransferF2C(x_fine, x_coarse);
} // MultiDofMapStep::TransferF2C


void
MultiDofMapStep::
AddF2C (double fac, BaseVector const *x_fine, BaseVector *x_coarse) const
{
  GetPrimMap()->AddF2C(fac, x_fine, x_coarse);
} // MultiDofMapStep::AddF2C


void
MultiDofMapStep::
TransferC2F (BaseVector *x_fine, BaseVector const *x_coarse) const
{
  GetPrimMap()->TransferC2F(x_fine, x_coarse);
} // MultiDofMapStep::TransferC2F


void
MultiDofMapStep::
AddC2F (double fac, BaseVector *x_fine, BaseVector const *x_coarse) const
{
  GetPrimMap()->AddC2F(fac, x_fine, x_coarse);
} // MultiDofMapStep::AddC2F


bool
MultiDofMapStep::
CanConcatenate (shared_ptr<BaseDOFMapStep> other)
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
} // MultiDofMapStep::CanConcatenate


shared_ptr<BaseDOFMapStep>
MultiDofMapStep::
Concatenate (shared_ptr<BaseDOFMapStep> other)
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
} // MultiDofMapStep::Concatenate


shared_ptr<BaseDOFMapStep>
MultiDofMapStep::
PullBack (shared_ptr<BaseDOFMapStep> other)
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
} // MultiDofMapStep::PullBack


shared_ptr<BaseMatrix>
MultiDofMapStep::
AssembleMatrix (shared_ptr<BaseMatrix> mat) const
{
  return maps[0]->AssembleMatrix(mat);
} // MultiDofMapStep::AssembleMatrix


void
MultiDofMapStep::
PrintTo (std::ostream & os, string prefix) const
{
  os << prefix << " MultiDofMapStep with " << GetNMaps() << " maps " << endl;
  for (auto k : Range(GetNMaps())) {
    os << prefix << "  map # " << k << ":" << endl;
    GetMap(k)->PrintTo(os, prefix + "  ");
    os << endl;
  }
} // MultiDofMapStep::PrintTo


int MultiDofMapStep::
GetBlockSize () const
{
  return GetMap(0)->GetBlockSize();
} // MultiDifMapStep::GetBlockSize


int MultiDofMapStep::
GetMappedBlockSize () const
{
  return GetMap(0)->GetMappedBlockSize();
} // MultiDifMapStep::GetMappedBlockSize

/** END MultiDofMapStep **/

} // namespace amg


namespace amg
{

template class ProlMap<double>;
template class ProlMap<Mat<2,2,double>>;
template class ProlMap<Mat<3,3,double>>;

#ifdef ELASTICITY
template class ProlMap<Mat<1,3,double>>;
template class ProlMap<Mat<1,6,double>>;
template class ProlMap<Mat<2,3,double>>;
template class ProlMap<Mat<3,6,double>>;
template class ProlMap<Mat<6,6,double>>;
#endif // ELASTICITY

// TODO: these seem to be needed from SOMEWHERE in the stokes AMG,
//       probably one of the distach-BS calls. They should not be,
//       but instead of finding that spot I just instantiate it here.
template class ProlMap<Mat<2,1,double>>;
template class ProlMap<Mat<1,2,double>>;
template class ProlMap<Mat<3,1,double>>;
template class ProlMap<Mat<3,2,double>>;
#if MAX_SYS_DIM >= 4
template class ProlMap<Mat<4,1,double>>;
template class ProlMap<Mat<1,4,double>>;
#endif
#if MAX_SYS_DIM >= 5
template class ProlMap<Mat<5,1,double>>;
template class ProlMap<Mat<1,5,double>>;
#endif
#if MAX_SYS_DIM >= 6
template class ProlMap<Mat<6,1,double>>;
template class ProlMap<Mat<6,2,double>>;
#endif

// elasticity workaround, no idea why we need these symbols now all of a sudden
template class ProlMap<Mat<5,6,double>>;
template class ProlMap<Mat<6,5,double>>;

template class DynBlockProlMap<double>;
} // namespace amg

