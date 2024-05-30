#ifndef FILE_AMG_PC_STOKES_IMPL_HPP
#define FILE_AMG_PC_STOKES_IMPL_HPP

#include "base_factory.hpp"
#include "dyn_block.hpp"
#include "hdiv_stokes_pc.hpp"

#include "stokes_pc_impl.hpp"

// #include "hdiv_hdg_embedding_impl.hpp"
#include "universal_dofs.hpp"
#include "utils_sparseLA.hpp"
#include <elementtopology.hpp>

namespace amg
{

/** HDivStokesAMGPC **/

template<class TFACTORY>
HDivStokesAMGPC<TFACTORY> :: HDivStokesAMGPC (shared_ptr<BilinearForm>      blf,
                                              Flags                  const &flags,
                                              string                 const &name,
                                              shared_ptr<Options>           opts,
                                              shared_ptr<BaseMatrix>        weightMat)
  : StokesAMGPrecond<TMESH>(blf, flags, name, opts)
{
  if (weightMat != nullptr)
  {
    std::cout << " =========================================================== " << std::endl;
    std::cout << "  WARNING: WEIGHT-MATRIX IGNORED BY HDIV (NOT IMPLEMENTED)" << std::endl;
    std::cout << " =========================================================== " << std::endl;
  }

  _facetAuxInfo = make_shared<FacetAuxiliaryInformation>(blf->GetFESpace());

  this->InitializeOptions();
  auto const &O = static_cast<Options&>(*this->options);

  _hDivHDGEmbedding = make_unique<HDivHDGEmbedding>(*this, O.auxSpace);

  // hDivFES = my_dynamic_pointer_cast<HDivHighOrderFESpace>(blf->GetFESpace(),
  //             "Need an HDiv-space for HDivStokesAMG");

} // HDivStokesAMGPC(..)


template<class TFACTORY>
HDivStokesAMGPC<TFACTORY> :: HDivStokesAMGPC (shared_ptr<FESpace>            fes,
                                              Flags                        &flags,
                                              string                 const &name,
                                              shared_ptr<Options>           opts,
                                              shared_ptr<BaseMatrix>        weightMat)
  : StokesAMGPrecond<TMESH>(fes, flags, name, opts)
{
  if (weightMat != nullptr)
  {
    std::cout << " =========================================================== " << std::endl;
    std::cout << "  WARNING: WEIGHT-MATRIX IGNORED BY HDIV (NOT IMPLEMENTED)" << std::endl;
    std::cout << " =========================================================== " << std::endl;
  }

  _facetAuxInfo = make_shared<FacetAuxiliaryInformation>(fes);

  this->InitializeOptions();
  auto const &O = static_cast<Options&>(*this->options);

  _hDivHDGEmbedding = make_unique<HDivHDGEmbedding>(*this, O.auxSpace);

  // hDivFES = my_dynamic_pointer_cast<HDivHighOrderFESpace>(fes,
  //             "Need an HDiv-space for HDivStokesAMG");

  // does this really do what we want it to??
  // facetAuxInfo = make_shared<FacetAuxiliaryInformation>(hDivFES);
} // HDivStokesAMGPC(..)


// template<class TFACTORY>
// void
// HDivStokesAMGPC<TFACTORY> ::
// SetVectorsToPreserve (FlatArray<shared_ptr<BaseVector>> vectorsToPreserve)
// {
//   preservedVecs.SetSize(vectorsToPreserve.Size());

//   for(auto k : Range(preservedVecs))
//   {
//     preservedVecs[k]  = vectorsToPreserve[k]->CreateVector();
//     *preservedVecs[k] = *vectorsToPreserve[k];
//   }
// } // HDivStokesAMGPC :: SetVectorsToPreserve


template<class TFACTORY>
shared_ptr<BaseAMGPC::Options> HDivStokesAMGPC<TFACTORY> :: NewOpts ()
{
  return make_shared<Options>();
} // HDivStokesAMGPC::NewOpts


template<class TFACTORY>
void HDivStokesAMGPC<TFACTORY> :: SetDefaultOptions (BaseAMGPC::Options& O)
{
} // HDivStokesAMGPC::SetDefaultOptions


template<class TFACTORY>
void HDivStokesAMGPC<TFACTORY> :: SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix)
{
  static_cast<Options&>(O).SetFromFlags(this->bfa->GetFESpace(), finest_mat, flags, prefix);
} // HDivStokesAMGPC::SetOptionsFromFlags


template<class TFACTORY>
void HDivStokesAMGPC<TFACTORY> :: ModifyOptions (BaseAMGPC::Options & aO, const Flags & flags, string prefix)
{
  BaseStokesAMGPrecond::ModifyOptions(aO, flags, prefix);
  // aO.force_ass_flmat = true;
} // HDivStokesAMGPC::ModifyOptions


template<class TFACTORY>
shared_ptr<TopologicMesh>
HDivStokesAMGPC<TFACTORY>::
BuildInitialMesh ()
{
  static Timer t("HDivStokesAMGPC::BuildInitialMesh");

  RegionTimer rt(t);

  auto [ topMesh, fvd ] = this->BuildTopMesh();

  shared_ptr<TMESH> algMesh = this->BuildEmptyAlgMesh(std::move(topMesh), fvd);

  return algMesh;
} // HDivStokesAMGPC::BuildInitialMesh


template<class TFACTORY>
shared_ptr<TopologicMesh>
HDivStokesAMGPC<TFACTORY>::
BuildAlgMesh (shared_ptr<BlockTM> &&ttopMesh, FlatArray<FVDescriptor> fvd)
{
  throw Exception("HDivStokesAMGPC::BuildAlgMesh should not be called anymore (happens now in BuildEmbedding) !");

  static Timer t1("HDivStokesAMGPC::BuildAlgMesh - fill");

  shared_ptr<BlockTM> topMesh = ttopMesh;

  const auto & O = static_cast<Options&>(*options);

  auto &auxInfo = GetFacetAuxInfo();
  const auto & hDivUDofs = GetHDivUDofs();

  auto algMesh = my_dynamic_pointer_cast<TMESH>(topMesh, "HDivStokesAMGPC::BuildAlgMesh - TMESH");

  auto edges = algMesh->template GetNodes<NT_EDGE>();

  auto const &MA = this->GetMA();

  auto comm = MA.GetCommunicator();

  // !! WILL NOT WORK WITH COMPOUND !!
  Array<int> facetDOFs;
  auto getLOFacetDOF = [&](auto fnr) {
    this->GetFESpace().GetDofNrs(NodeId(NT_FACET, fnr), facetDOFs);
    return facetDOFs[0];
  };

  // weights
  switch(O.energy) {
    case(Options::TRIV_ENERGY):
    {
      FillAlgMesh(*algMesh,
                  [&](auto elnum, auto fnum, auto const &A) { return 1.0; });

      break;
    }
    case(Options::ALG_ENERGY):
    {
      FillAlgMesh(*algMesh,
                  [&](auto elnum, auto fnum, auto const &A) {
        // maximum (by abs-of-trace) connection between facet "fnum" and any other facet in the same element
        auto di = getLOFacetDOF(fnum);
        // auto di = auxInfo.A2R_Facet(fnum);
        auto facets0 = this->GetMA().GetElFacets(ElementId(VOL, elnum));
        double w0 = 0;
        for (auto fnumj : facets0) {
          if (fnumj != fnum) {
            auto dj = getLOFacetDOF(fnumj);
            // int dj = auxInfo.A2R_Facet(fnumj);
            if (dj != -1) { // I think this cannot happen per definition (is a facet of an element!)
              // cout << " dj " << dj << endl;
              // w0 = max2(w0, fabs(calc_trace(A(di, dj))));
              auto const aij = fabs(calc_trace(A(di, dj)));
              auto const aii = fabs(calc_trace(A(di, di)));
              auto const ajj = fabs(calc_trace(A(dj, dj)));
              w0 = max2(w0, fabs(aij / sqrt(aii * ajj)));
            }
          }
        }
        return w0;
      });

      break;
    }
    case(Options::ELMAT_ENERGY):
    {
      throw Exception("Stokes elmat energy not implemented!");
      break;
    }
    default:
    {
      throw Exception("Stokes Invalid Energy!");
      break;
    }
  }

  return algMesh;
} // HDivStokesAMGPC::BuildAlgMesh


template<class TFACTORY>
template<class TLAM>
void HDivStokesAMGPC<TFACTORY> :: FillAlgMesh(TMESH const &alg_mesh, TLAM weightLambda)
{
  // TODO
  auto &auxInfo = GetFacetAuxInfo();

  shared_ptr<BaseMatrix> wmat = (finest_weight_mat == nullptr) ? finest_mat : finest_weight_mat;
  shared_ptr<SparseMatrix<double>> finest_spm;

  if (auto pmat = dynamic_pointer_cast<ParallelMatrix>( wmat ))
    { finest_spm = dynamic_pointer_cast<SparseMatrix<double>>(pmat->GetMatrix()); }
  else
    { finest_spm = dynamic_pointer_cast<SparseMatrix<double>>(wmat); }

  if (finest_spm == nullptr)
    { throw Exception("No valid matrix in HDivStokesAMGPC::FillAlgMesh_ALG!"); }

  auto const &fes = this->GetFESpace();
  auto const &MA = this->GetMA();
  // auto ma = this->bfa->GetFESpace()->GetMeshAccess();

  auto edges = alg_mesh.template GetNodes<NT_EDGE>();
  auto aed   = get<1>(alg_mesh.Data());
  auto edata = aed->Data();

  const auto & A = *finest_spm;

  Array<int> elnums;

  for (auto ffnr : Range(auxInfo.GetNFacets_R()))
  {

    auto const  fnr      = auxInfo.R2A_Facet(ffnr);
    auto const  enr      = F2E(fnr);
    auto const &edge     = edges[enr];

    MA.GetFacetElements(fnr, elnums);

    double wij = 0.0;
    double wji = 0.0;

    // cout << " FillAlgMesh, fnr " << fnr << "-> enr " << endl;

    if (fes.DefinedOn(ElementId(VOL, elnums[0])))
      { wij = weightLambda(elnums[0], fnr, A); }

    if ( (elnums.Size() == 2)  && fes.DefinedOn(ElementId(VOL, elnums[1])) )
      { wji = weightLambda(elnums[1], fnr, A); }

    bool const flipped = EL2V(elnums[0]) == edge.v[1];

    if (flipped)
      { swap(wij, wji); }

    auto nc_dps = hDivUDofs.GetDistantProcs(ffnr);

    if ( nc_dps.Size() ) {
      // flipped:     v1 with wt wji is local
      // not flipped: v0 with wt wij is local
      edata[enr].edi = flipped ? 0.0 : wij;
      edata[enr].edj = flipped ? wji : 0.0;
    }
    else {
      edata[enr].edi = wij;
      edata[enr].edj = wji;
    }
  }
}


template<class TFACTORY>
Array<INT<2, double>> HDivStokesAMGPC<TFACTORY> :: CalcFacetFlows () const
{
  throw Exception("HDivStokesAMGPC::CalcFacetFlows should not be called anymore!");

  /**
   * Flow is oriented from lower element to higher element. ( = outside w.r.t smallest element )
   * So also oriented outside the domain.
   * So INCOSISTENT for MPI-BNDS ! (that is OK, I have to flip it later!)
   */
  LocalHeap lh(10 * 1024 * 1024, "Orrin_Oriscorin_Hiscorin_Sincorin_Alvin_Vladimir_Groolmoplong");
  // auto f2a_facet = hDivFES->GetFMapF2A();

  auto fes      = this->GetBilinearForm()->GetFESpace();
  auto ma       = fes->GetMeshAccess();
  auto &auxInfo = GetFacetAuxInfo();

  auto const fesDim = fes->GetDimension();

  // CoefficientFunction for u*n
  shared_ptr<CoefficientFunction> trial_u
    = make_shared<ProxyFunction>(
        this->bfa->GetFESpace(),                    // fespace
        false,                      // testfunction
        false,                      // complex
        this->bfa->GetFESpace()->GetEvaluator(VOL), // evaluator
        nullptr,                    // deriv_evaluator
        nullptr,                    // trace_evaluator
        nullptr,                    // trace_deriv_evaluator
        nullptr,                    // ttrace_evaluator
        nullptr                     // ttrace_deriv_evaluator
      );
  shared_ptr<CoefficientFunction> normal = NormalVectorCF(this->GetMA().GetDimension());
  shared_ptr<CoefficientFunction> u_dot_n = InnerProduct(trial_u, normal);

  // Integrator
  auto lint =
    make_shared<SymbolicLinearFormIntegrator>
      (u_dot_n,      // cf
       VOL,          // VorB
       BND);         // element-VorB

  // extra integration order for curved elements (should suffice I hope)
  int curve_order = this->GetMA().GetCurveOrder(); // TODO: get this from mesh
  int ir_order    = 2 + this->GetMA().GetDimension() * curve_order;

  lint->SetBonusIntegrationOrder(this->GetMA().GetDimension() * curve_order);

  Array<INT<2, double>> flows(auxInfo.GetNFacets_R());

  BitArray facetDone(auxInfo.GetNFacets_R());
  facetDone.Clear();

  // Array<int> elFacets(50);
  Array<int> facetDofs(50);
  Array<int> elDOFs(50);
  Array<int> facetEls(50);

  for (auto relnr : Range(auxInfo.GetNE_R()))
  {
    HeapReset hr(lh);

    auto elnr = auxInfo.R2A_EL(relnr);

    ElementId eid(VOL, elnr);

    // this->GetMA().GetElFacets(eid, elFacets); // deprecated ?
    auto elFacets = this->GetMA().GetElFacets(eid);

    bool anyNew = false;

    for (auto fnum : elFacets) {
      auto fnum_R = auxInfo.A2R_Facet(fnum);
      if ( (fnum_R != -1) && (!facetDone.Test(fnum_R)) )
        { anyNew = true; }
    }

    if (!anyNew)
      { continue; }

    auto const &fel   = fes->GetFE(eid, lh);
    auto const &elTrans = this->GetMA().GetTrafo(eid, lh);
    ELEMENT_TYPE et_vol = elTrans.GetElementType();

    fes->GetDofNrs(eid, elDOFs);

    FlatVector<double> elvec(elDOFs.Size() * fesDim, lh);

    lint->CalcElementVector
            (fel,
             elTrans,
             elvec,
             lh);

    for (auto loc_facet_nr : Range(elFacets))
    {
      auto const fnum = elFacets[loc_facet_nr];
      auto const fnum_R = auxInfo.A2R_Facet(fnum);

      if ( (fnum_R != -1) && (!facetDone.Test(fnum_R)) )
      {
        NodeId facetId(NT_FACET, fnum);

        fes->GetDofNrs(facetId, facetDofs);
        this->GetMA().GetFacetElements(facetId, facetEls);

        auto const loFacetDOF = facetDofs[0];

        if (loFacetDOF == -1)
        {
          // should not happen - any unused facet should not be R!
          throw Exception("LO-DOF is unused on R-facet!?");
        }

        int  const pos = elDOFs.Pos(loFacetDOF);

        // we are computing the flow from elnr outwards
        // we want it oriented facet-el 0 -> facet-el 1
        double const orientationFactor = ( facetEls[0] == elnr ) ? 1.0 : -1.0;

        facetDone.SetBit(fnum_R);
        flows[fnum_R][0] = orientationFactor * elvec[pos];


        ELEMENT_TYPE et_facet = ElementTopology::GetFacetType (et_vol, loc_facet_nr);
        const IntegrationRule & ir_facet = SelectIntegrationRule (et_facet, ir_order); // reference facet
        Facet2ElementTrafo facet_2_el(et_vol, BND); // reference facet -> reference vol
        IntegrationRule & ir_vol = facet_2_el(loc_facet_nr, ir_facet, lh); // reference VOL
        BaseMappedIntegrationRule &baseMIR = elTrans(ir_vol, lh);

        // double facetSurf = 0.0;
        // for (auto const &mip : baseMIR)
        // {
        //   facetSurf += mip.GetWeight();
        // }
        // flows[fnum_R][1] = facetSurf;

        flows[fnum_R][1] = std::accumulate(baseMIR.begin(), baseMIR.end(), 0.0,
          [&](auto const &partialSum, BaseMappedIntegrationPoint const &mip) { return partialSum + mip.GetWeight(); });

        cout << endl << " EID " << eid << endl;
        cout << " facetDofs "; prow2(facetDofs); cout << endl;
        cout << " elDOFs "; prow2(elDOFs); cout << endl;
        cout << " elVec "; prow2(elvec); cout << endl;
        cout << "    LO-DOF " << loFacetDOF << " at " << pos << endl;
        cout << " fnum_R " << fnum_R << ", facet-id " << fnum << " start at " << facetEls[0] << " w. flow " << flows[fnum_R][0] << ", surf " << flows[fnum_R][1] << endl;

      }
    }
  }

  return flows;
} // HDivStokesAMGPC::CalcFacetFlows


template<class TFACTORY>
BaseAMGFactory&
HDivStokesAMGPC<TFACTORY>::
GetBaseFactory () const
{
  return GetFactory();
} // VertexAMGPC::GetBaseFactory


template<class TFACTORY>
SecondaryAMGSequenceFactory const&
HDivStokesAMGPC<TFACTORY>::
GetSecondaryAMGSequenceFactory () const
{
  return GetFactory();
} // VertexAMGPC::GetSecondaryAMGSequenceFactory


template<class TFACTORY>
TFACTORY&
HDivStokesAMGPC<TFACTORY>::
GetFactory () const
{
  if (_factory == nullptr)
  {
    auto opts = dynamic_pointer_cast<Options>(options);
   _factory =  make_shared<TFACTORY>(opts);
  }
  return *_factory;
} // VertexAMGPC::GetFactory


template<class TFACTORY>
shared_ptr<BaseDOFMapStep> HDivStokesAMGPC<TFACTORY> :: BuildEmbedding (BaseAMGFactory::AMGLevel &level)
{
  // TODO
  /**
   * We need a multi-embedding here:
   *   i) mesh edge-space      -> fes
   *  ii) mesh potential-space -> fes
   *        this is just a concatenation of the curl-matrix and the edge->hDivFES embedding from (i)
   *
   * Note: We need to take into account the difference between the facet-midpoint and vertex-midpoint position
   *       This does not matter for "BND"-vertices; Their pos is set s.t facet-MP is exactly the vertex-MP
   */
  auto const &O = *options;

  auto &cap = *my_dynamic_pointer_cast<typename TFACTORY::HDivStokesLevelCapsule>(level.cap,
                  "HDivStokesAMGPC::InitFinestLevel - wrong CAP type!");

  auto fMesh  = my_dynamic_pointer_cast<TMESH>(cap.mesh,
                  "HDivStokesAMGPC::InitFinestLevel - wrong mesh type!");

  // HDivHDGEmbedding E(*this);

  auto &E = *_hDivHDGEmbedding;

  cap.meshDOFs = E.CreateMeshDOFs(fMesh);

  if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG)
  {
    auto const rk = fMesh->GetEQCHierarchy()->GetCommunicator().Rank();
    auto const fn = "stokes_mesh_dofs_rk_" + std::to_string(rk) + "_l_0.out";
    std::ofstream of(fn);
    cap.meshDOFs->template PrintAs<TMESH>(of);
  }

  auto const &meshDOFs = *cap.meshDOFs;

  UniversalDofs const meshUDofs = GetFactory().BuildUDofs(cap);
  UniversalDofs const fesUDofs  = MatToUniversalDofs(*finest_mat);

  auto [rangeProl, facetSurf, facetFlows, presVecs]
    = E.CreateDOFEmbedding(*fMesh,
                           *cap.meshDOFs);

  auto rangeEmb =  make_shared<ProlMap<double>>
                      (rangeProl,
                       fesUDofs,
                       meshUDofs);

  // cap.preservedVectors  = E.CreatePreservedVectors(*level.embed_map);

  // cap.preservedVectors  = BuildPreservedVectors(rangeEmb.get());

  cap.preservedVectors = make_shared<PreservedVectors>(1, std::move(presVecs));

  // cout << " rangeEmb: " << endl << *rangeEmb << endl;

  /** set facet-flows as computed in CreateDOFEmbedding **/

  auto const &auxInfo = GetFacetAuxInfo();
  auto const &MA      = *E.GetFESpace().GetMeshAccess();

  auto comm = MA.GetCommunicator();

  auto &attEdgeData = *get<1>(fMesh->Data());
  auto edgeData = attEdgeData.Data();

  attEdgeData.SetParallelStatus(DISTRIBUTED); // just to be sure

  auto const &FM = *fMesh;

  // cout << " NOW fill FLOW" << endl;
  // iterate through "solid" edges where we have a local element
  for (auto ffnr : Range(auxInfo.GetNFacets_R())) {

    auto const  fnr  = auxInfo.R2A_Facet(ffnr);
    auto const  enr  = F2E(fnr);
    auto const &edge = FM.template GetNode<NT_EDGE>(enr);

    auto facetDPs = MA.GetDistantProcs(NodeId(NT_FACET, fnr));

    edgeData[enr].flow    = 0;

    // only set the flow on 1 rank!
    if ( (facetDPs.Size() == 0) ||
         (comm.Rank() < facetDPs[0]) )
    {
      edgeData[enr].flow[0] = facetFlows[fnr];
    }
  }

  // cout << " NOW fill weights" << endl;
  switch(O.energy) {
    case(Options::TRIV_ENERGY):
    {
      FillAlgMesh(FM,
                  [&](auto elnum, auto fnum, auto const &A) { return 1.0; });

      break;
    }
    case(Options::ALG_ENERGY):
    {
      FillAlgMesh(FM,
                  [&](auto elnum, auto fnum, auto const &A)
      {
        // maximum (by abs-of-trace) connection between facet "fnum" and any other facet in the same element
        // cout << "FiillAlgMesh LAM, elnum = " << elnum << endl;
        auto di = E.GetLOFacetDOF(fnum);
        // cout << " -> di = " << di << endl;
        // auto di = auxInfo.A2R_Facet(fnum);
        auto facets0 = this->GetMA().GetElFacets(ElementId(VOL, elnum));
        double w0 = 0;
        for (auto fnumj : facets0)
        {
          if (fnumj != fnum)
          {
            auto dj = E.GetLOFacetDOF(fnumj);
            // cout << " -> dj = " << dj << endl;

            if (dj != -1) { // I think this cannot happen per definition (is a facet of an element!)
              // cout << " dj " << dj << endl;
              // w0 = max2(w0, fabs(calc_trace(A(di, dj))));
              auto const aij = fabs(calc_trace(A(di, dj)));
              auto const aii = fabs(calc_trace(A(di, di)));
              auto const ajj = fabs(calc_trace(A(dj, dj)));
              w0 = max2(w0, fabs(aij / sqrt(aii * ajj)));
            }
          }
        }
        return w0;
      });
      break;
    }
    case(Options::ELMAT_ENERGY): { throw Exception("Stokes elmat energy not implemented!");  break; }
    default:                     { throw Exception("Stokes Invalid Energy!");          break; }
  }

  /** embedding mesh-potential -> NC-space **/

  GetFactory().BuildCurlMat(cap);
  GetFactory().BuildPotUDofs(cap);

  // TODO: AAARGGS I HATE IT
  auto &curlMat = static_cast<typename TFACTORY::TCM_TM &>(*cap.curl_mat);

  // cout << " CHECK CM ETRS " << endl;
  // for (auto k : Range(cmat.Height())) {
  //   auto ris = cmat.GetRowIndices(k);
  //   auto rvs = cmat.GetRowValues(k);
  //   for (auto j : Range(ris)) {
  //     if (ris[j] == -1) {
  //       cout << " CM ETR -1 in row " << k << ", val = " << rvs[j] << endl;
  //       cout << " all ris = "; prow(ris); cout << endl << endl;
  //     }
  //   }
  // }
  // cout << " DONE CHECK CM ETRS " << endl;

  // if (M.GetEQCHierarchy()->GetCommunicator().Rank() == 35) {
    // cout << " curl_mat: " << cmat.Height() << " x " << cmat.Width() << endl;
    // print_tm_spmat(cout << endl << endl << endl, cmat);
  // }

  // make sure the SparseMatrixTM MatMultAB instantiation is used (not SparseMatrix<double>)
  SparseMatrixTM<double> &rangeProlTM(*rangeProl);
  auto potProl = MatMultAB(rangeProlTM, curlMat);

  auto potEmb = make_shared<ProlMap<double>> (potProl, fesUDofs, cap.pot_uDofs);
  // emb_pot->Finalize();

  // if (M.GetEQCHierarchy()->GetCommunicator().Rank() == 35) {
  //   cout << endl << endl << " final emb mat " << endl; print_tm_spmat(cout << endl, *cep); cout << endl;
  //   cout << endl << endl << " final mat / emb dims " << finest_mat->Height() << " x " << finest_mat->Width() << " times " <<
  // 	cep->Height() << " x " << cep->Width() << endl;
  // }

  /** combined **/
  Array<shared_ptr<BaseDOFMapStep>> steps( { rangeEmb, potEmb } );

  auto embStep = make_shared<MultiDofMapStep>(steps);

  embStep->Finalize();

  if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG)
  {
    int const rk = meshUDofs.GetCommunicator().Rank();
    std::string ofn = "ngs_amg_embedding_rk_" + std::to_string(rk) + ".out";
    std::ofstream ofs(ofn);
    embStep->PrintTo(ofs);
  }

  return embStep;
} // HDivStokesAMGPC::BuildEmbedding


template<class TFACTORY>
shared_ptr<MeshDOFs>
HDivStokesAMGPC<TFACTORY> ::
BuildMeshDOFs (shared_ptr<TMESH> const &fMesh)
{
  throw Exception("HDivStokesAMGPC::BuildMeshDOFs should not be called anymore!");

  auto const &O = *options;

  // On mesh-level "0", do we have exactly 1 or 2 DOFs per facet?
  // I think we have 1, only the lowest order, RT, DOF should contribute.
  int const nDOFPerFacet    = 1;

  int NE = fMesh->template GetNN<NT_EDGE>();

  auto meshDOFs = make_shared<MeshDOFs>(fMesh);

  cout << " FINEST meshDOFs = " << endl;
  meshDOFs->template PrintAs<TMESH>(cout);
  cout << endl;

  auto [dofed_edges, dof2e, e2dof] = fMesh->GetDOFedEdges();

  Array<int> offsets(NE + 1);
  offsets[0] = 0;
  for (auto k : Range(NE))
  {
    // gg-edges have no DOFs
    offsets[k + 1] = offsets[k] + (dofed_edges->Test(k) ? nDOFPerFacet : 0);
  }

  meshDOFs->SetOffsets(std::move(offsets));

  // TODO: this should be printed from the factory based on factory-log-level somewhere
  if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG)
  {
    auto const rk = fMesh->GetEQCHierarchy()->GetCommunicator().Rank();
    auto const fn = "stokes_mesh_dofs_rk_" + std::to_string(rk) + "_l_0.out";
    std::ofstream of(fn);
    meshDOFs->template PrintAs<TMESH>(of);
  }

  return meshDOFs;
} // HDivStokesAMGPC :: BuildMeshDOFs


// template<class TFACTORY>
// shared_ptr<PreservedVectors>
// HDivStokesAMGPC<TFACTORY> ::
// BuildPreservedVectors(BaseDOFMapStep const *embedding)
// {
//   // TODO: We should get the POST-embedded vecs from HDivHDGEmbedding,
//   //       that would be much easier!
//   if (embedding == nullptr)
//   {
//     throw Exception("BuildPreservedVectors without embedding!");
//   }

//   int const nSpecial   = 1;
//   int const nPreserved = preservedVecs.Size();

//   Array<shared_ptr<BaseVector>> meshVecs(nPreserved);

//   for (auto k : Range(meshVecs))
//   {
//     meshVecs[k] = embedding->CreateMappedVector();
//     embedding->TransferF2C(preservedVecs[k].get(), meshVecs[k].get());
//   }

//   return make_shared<PreservedVectors>(nSpecial, std::move(meshVecs));
// } // HDivStokesAMGPC :: BuildPreservedVectors


template<class TFACTORY>
void HDivStokesAMGPC<TFACTORY> :: InitFinestLevel (BaseAMGFactory::AMGLevel & finestLevel)
{
  // TODO
  auto & O(static_cast<Options&>(*options));
  if (O.force_ass_flmat)
    { throw Exception("force_ass_flmat for stokes probably not correct yet"); }

  // BaseAMGPC::InitFinestLevel does stuff that needs meshDOFs already
  // so we cannt call it here
  // BaseAMGPC::InitFinestLevel(finestLevel);

  finestLevel.level = 0;
  finestLevel.cap = GetFactory().AllocCap();

  auto &cap = *my_dynamic_pointer_cast<typename TFACTORY::HDivStokesLevelCapsule>(finestLevel.cap,
                  "HDivStokesAMGPC::InitFinestLevel - wrong CAP type!");

  cap.mesh             = this->BuildInitialMesh(); // TODO: get out of factory??

  auto fMesh  = my_dynamic_pointer_cast<TMESH>(cap.mesh,
                  "HDivStokesAMGPC::InitFinestLevel - wrong mesh type!");

  cap.eqc_h             = cap.mesh->GetEQCHierarchy();

  finestLevel.embed_map = BuildEmbedding(finestLevel);

  cap.mat = GetLocalSparseMat(finest_mat);
  cap.embeddedRangeMatrix = my_dynamic_pointer_cast<typename TFACTORY::TSPM>
                              (finestLevel.embed_map->AssembleMatrix(cap.mat),
                               "HDivStokesAMGPC::InitFinestLevel - embedded R-matrix!");

  if (finestLevel.embed_map == nullptr) {
    throw Exception("Should never get here for HDIV!");

    cap.uDofs = MatToUniversalDofs(*finest_mat, DOF_SPACE::ROWS);
  }
  else {
    /** Explicitely assemble matrix associated with the finest mesh. **/
    if (options->force_ass_flmat) {
      cap.mat = cap.embeddedRangeMatrix;
      finestLevel.embed_done = true;
    }
    /** Either way, pardofs associated with the mesh are the mapped pardofs of the embed step **/
    cap.uDofs = finestLevel.embed_map->GetMappedUDofs();
  }

  /** Dirichlet Conditions **/

  auto floops = fMesh->GetLoops();

  auto [free_verts, free_edges, free_psn] = this->BuildFreeNodes(*fMesh, floops, fMesh->GetLoopUDofs());

  cap.pot_freedofs = free_psn;   // potential space smoother freedofs
  cap.free_nodes   = free_verts; // coarsening "freedofs"
  fMesh->SetFreeNodes(free_edges);        // for prolongation
} // HDivStokesAMGPC::InitFinestLevel


// template<class TFACTORY>
// bool
// HDivStokesAMGPC<TFACTORY>::
// SupportsBlockSmoother(const BaseAMGFactory::AMGLevel &aMGLevel)
// {
//   return true;
// }


template<class TFACTORY>
Table<int>
HDivStokesAMGPC<TFACTORY>::
GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
{
  // TODO: add option to do a larger block-smoother with blocks:
  //          * one block for every coarse edge containing all DOFs in fine edges mapping to it
  //          * one block consisting of interior DOFs of every agglommerate

  if (amg_level.level == 0)
  {
    return GetFinestLevelGSBlocks();
  }

  auto &cap = *my_dynamic_pointer_cast<typename TFACTORY::HDivStokesLevelCapsule>(amg_level.cap,
                  "HDivStokesAMGPC::GetGSBlocks - wrong CAP type!");

  auto const &TM = *my_dynamic_pointer_cast<TMESH>(cap.mesh,
                  "HDivStokesAMGPC::GetGSBlocks - wrong mesh type!");

  auto const &meshDOFs = *cap.meshDOFs;

  // 1 Block per edge
  TableCreator<int> createBlocks;

  int cntBlocks = 0;
  for (; !createBlocks.Done(); createBlocks++)
  {
    cntBlocks = 0;

    for (auto k : Range(TM.template GetNN<NT_EDGE>()))
    {
      auto edgeDOFs = meshDOFs.GetDOFNrs(k);

      if (edgeDOFs.Size())
      {
        createBlocks.Add(cntBlocks++, edgeDOFs);
      }
    }
  }

  return createBlocks.MoveTable();
} // HDivStokesAMGPC::GetGSBlocks


template<class TFACTORY>
Table<int>
HDivStokesAMGPC<TFACTORY>::
GetFinestLevelGSBlocks ()
{
  TableCreator<int> createBlocks;

  auto const &fes     = _hDivHDGEmbedding->GetFESpace();
  auto const &MA      = *fes.GetMeshAccess();
  auto const &auxInfo = GetFacetAuxInfo();

  BitArray const *freedofs = this->finest_freedofs.get();

  unsigned cntBlocks = 0;

  Array<int> dofs;
  Array<int> validFreeDOFs;

  auto addBlock = [&](FlatArray<int> dofs)
  {
      validFreeDOFs.SetSize(dofs.Size());

      size_t blockSize = 0;

      for (auto dof : dofs)
      {
        if (dof >= 0) // defon, compress, etc
        {
          if (!freedofs || freedofs->Test(dof)) // Dirichlet, elint, etc
          {
            validFreeDOFs[blockSize++] = dof;
          }
        }
      }

      if (blockSize)
      {
        createBlocks.Add(cntBlocks++, validFreeDOFs.Range(0ul, blockSize));
      }
  };

  for(; !createBlocks.Done(); createBlocks++)
  {
    cntBlocks = 0;

    // 1 Block per facet
    for (auto facetNr : Range(MA.GetNFacets()))
    {
      fes.GetDofNrs(NodeId(NT_FACET, facetNr), dofs);
      addBlock(dofs);
    }

    // 1 block per element (if order is high enough order and no elint)
    for (auto elNr: Range(MA.GetNE()))
    {
      fes.GetDofNrs(NodeId(NT_ELEMENT, elNr), dofs);
      addBlock(dofs);
    }
  }

  return createBlocks.MoveTable();
} // HDivStokesAMGPC::GetGSBlocks


template<class TFACTORY>
void HDivStokesAMGPC<TFACTORY> :: FinalizeLevel (shared_ptr<BaseMatrix> mat)
{
  hDivUDofs = MatToUniversalDofs(*mat, DOF_SPACE::COLS);

  StokesAMGPrecond<TMESH>::FinalizeLevel(mat);
} // HDivStokesAMGPC::FinalizeLevel


template<class TFACTORY>
FacetAuxiliaryInformation const & HDivStokesAMGPC<TFACTORY> :: GetFacetAuxInfo () const
{
  if ( _facetAuxInfo == nullptr )
    { throw Exception("FACET-Aux info not set up in HDivStokesAMGPC! "); }
  // { const_cast<shared_ptr<FacetAuxiliaryInformation>&>(facetAuxInfo) = make_shared<FacetAuxiliaryInformation>(this->bfa->GetFESpace()); }
  return *_facetAuxInfo;
} // HDivStokesAMGPC::SetFacetAuxInfo


template<class TFACTORY>
Array<shared_ptr<BaseSmoother>>
HDivStokesAMGPC<TFACTORY>::
BuildSmoothers (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> levels,
                shared_ptr<DOFMap> dof_map)
{
  auto const &O = static_cast<Options&>(*options);

  if (O.log_level == Options::LOG_LEVEL::DBG)
  {
    auto const nLevels = levels.Size();

    stashedPresVecs.SetSize(nLevels);

    for (auto k : Range(nLevels))
    {
      auto const &cap = static_cast<typename TFACTORY::HDivStokesLevelCapsule&>(*levels[k]->cap);
      auto const &presVecs = *cap.preservedVectors;

      stashedPresVecs[k].SetSize(presVecs.GetNPreserved());

      for (auto j : Range(presVecs.GetNPreserved()))
      {
        stashedPresVecs[k][j] = presVecs.GetVectorPtr(j);
      }
    }

    stashedEmb = levels[0]->embed_map;
  }
  else
  {
    stashedPresVecs.SetSize0();
    stashedEmb = nullptr;
  }

  return StokesAMGPrecond<TMESH>::BuildSmoothers(levels, dof_map);
} // HDivStokesAMGPC::BuildSmoothers


template<class TFACTORY>
void
HDivStokesAMGPC<TFACTORY>::
OptimizeDOFMap (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> aMGLevels,
                shared_ptr<DOFMap> dOFMap)
{
  auto & O(static_cast<Options&>(*options));

  if ( ( dOFMap->GetNSteps() == 0 ) ||
       ( O.use_dynbs_prols == false ) )
  {
    return;
  }

  auto convertDMS = [&](shared_ptr<BaseDOFMapStep> initStep,
                        DynVectorBlocking<>        *fBlocking,
                        DynVectorBlocking<>        *cBlocking) -> shared_ptr<BaseDOFMapStep>
  {

    auto convertProlMap = [&](shared_ptr<ProlMap<double>> prolStep) -> shared_ptr<BaseDOFMapStep>
    {
      shared_ptr<DynBlockSparseMatrix<double>> dynP, dynPT;

      if (fBlocking != nullptr)
      {
        dynP = make_shared<DynBlockSparseMatrix<double>>(*prolStep->GetProl(),
                                                         *fBlocking,
                                                         *cBlocking,
                                                         false);

        // cout << " SPARSE P I" << endl;
        // prolStep->GetProl()->Print(cout);
        // cout << endl;

        // cout << " dynP I" << endl;
        // dynP->PrintTo(cout);

        // cout << " SPARSE PT I" << endl;
        // prolStep->GetProlTrans()->Print(cout);
        // cout << endl;

        dynPT = make_shared<DynBlockSparseMatrix<double>>(*prolStep->GetProlTrans(),
                                                          *cBlocking,
                                                          *fBlocking,
                                                          false);
        // cout << " dynPT I" << endl;
        // dynPT->PrintTo(cout);
      }
      else
      {
        // row-blocking from sparse-mat graph
        dynP = make_shared<DynBlockSparseMatrix<double>>(*prolStep->GetProl(),
                                                         cBlocking); // <- PTR, NOT REF!

        // cout << " SPARSE P II" << endl;
        // prolStep->GetProl()->Print(cout);
        // cout << endl;

        // cout << " dynP II" << endl;
        // dynP->PrintTo(cout);

        dynPT = make_shared<DynBlockSparseMatrix<double>>(*prolStep->GetProlTrans(),
                                                          *cBlocking,
                                                          dynP->GetRowBlocking(),
                                                          false);

        // cout << " dynPT II" << endl;
        // dynPT->PrintTo(cout);
      }

      return make_shared<DynBlockProlMap<double>>(dynP, dynPT, prolStep->GetUDofs(), prolStep->GetMappedUDofs());
    };

    if (auto prolStep = dynamic_pointer_cast<ProlMap<double>>(initStep))
    {
      return convertProlMap(prolStep);
    }
    else if (auto concStep = dynamic_pointer_cast<ConcDMS>(initStep))
    {
      shared_ptr<BaseDOFMapStep> firstStep = concStep->GetStep(0);
    
      if (auto prolStep = dynamic_pointer_cast<ProlMap<double>>(firstStep))
      {
        Array<shared_ptr<BaseDOFMapStep>> newSteps(concStep->GetNSteps());
        newSteps[0] = convertProlMap(prolStep);

        if ( concStep->GetNSteps() > 1 )
        {
          for (auto k : Range(1, concStep->GetNSteps()))
          {
            newSteps[k] = concStep->GetStep(k);
          }
        }

        return make_shared<ConcDMS>(newSteps);
      }
      else
      {
        throw Exception("OptimizeDOFMap failed - not sure what to do here!");
        return prolStep;
      }
    }
    else
    {
      throw Exception("OptimizeDOFMap failed - not sure what to do here!");
      return prolStep;
    }
  };
  
  /**
   *  level 0: row-blocking: compute from sparsity (blocking of FES, not mesh!)
   *           col-blocking: pre-ctr mesh l_1 ( cna be != l_1-mesh  )
   */

  auto &finestCap = static_cast<typename TFACTORY::HDivStokesLevelCapsule&>(*aMGLevels[0]->cap);

  auto newFirstStep = convertDMS(dOFMap->GetStep(0),
                                 nullptr,
                                 &finestCap.preCtrCDOFBlocking);

  dOFMap->ReplaceStep(0, newFirstStep);

  if (dOFMap->GetNSteps() > 1)
  {
    for (auto k : Range(1ul, dOFMap->GetNSteps()))
    {
      /**
       *  level k: row-blocking: l_k-mesh
       *           col-blocking: pre-ctr l_(k+1)-mesh
       */

      auto &fCap = static_cast<typename TFACTORY::HDivStokesLevelCapsule&>(*aMGLevels[k]->cap);

      // cout << "convertDMS " << k << " -> " << k+1 << endl;
      auto newStep = convertDMS(dOFMap->GetStep(k),
                                &fCap.dOFBlocking,
                                &fCap.preCtrCDOFBlocking);

      dOFMap->ReplaceStep(k, newStep);
    }
  }
} // HDivStokesAMGPC::OptimizeDOFMap



/** END HDivStokesAMGPC **/

} // namespace amg


#endif // FILE_AMG_PC_STOKES_IMPL_HPP
