#ifndef FILE_NC_STOKES_PC_IMPL_HPP
#define FILE_NC_STOKES_PC_IMPL_HPP

#include "nc_stokes_pc.hpp"
#include "stokes_pc_impl.hpp"

namespace amg
{
extern void DoTest (BaseMatrix &mat, BaseMatrix &pc, NgMPI_Comm & gcomm, string message);


/** NCStokesAMGPC **/


template<class TFACTORY>
NCStokesAMGPC<TFACTORY> :: NCStokesAMGPC  (shared_ptr<BilinearForm>        blf,
                                           Flags                    const &flags,
                                           string                   const &name,
                                           shared_ptr<Options>             opts,
                                           shared_ptr<BaseMatrix>          weight_mat)
 : StokesAMGPrecond<TMESH>(blf, flags, name, opts)
 , finest_weight_mat(weight_mat)
 , nc_fes(my_dynamic_pointer_cast<NoCoH1FESpace>(blf->GetFESpace(), "NCStoeksAMGPC - NC-SPACE!"))
{
  ;
} // NCStokesAMGPC(..)


template<class TFACTORY>
NCStokesAMGPC<TFACTORY> :: NCStokesAMGPC (shared_ptr<FESpace>           fes,
                                          Flags                        &flags,
                                          string                 const &name,
                                          shared_ptr<Options>           opts,
                                          shared_ptr<BaseMatrix>        weight_mat)
  : StokesAMGPrecond<TMESH>(fes, flags, name, opts)
  , finest_weight_mat(weight_mat)
  , nc_fes(my_dynamic_pointer_cast<NoCoH1FESpace>(fes, "NCStoeksAMGPC - NC-SPACE!"))
{
  ;
} // NCStokesAMGPC(..)



template<class TFACTORY>
shared_ptr<BaseAMGPC::Options> NCStokesAMGPC<TFACTORY> :: NewOpts ()
{
  return make_shared<Options>();
} // NCStokesAMGPC::NewOpts


template<class TFACTORY>
void NCStokesAMGPC<TFACTORY> :: SetDefaultOptions (BaseAMGPC::Options& O)
{
} // NCStokesAMGPC::SetDefaultOptions


template<class TFACTORY>
void NCStokesAMGPC<TFACTORY> :: SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix)
{
  static_cast<Options&>(O).SetFromFlags(this->GetFESpacePtr(), finest_mat, flags, prefix);
} // NCStokesAMGPC::SetOptionsFromFlags


template<class TFACTORY>
void NCStokesAMGPC<TFACTORY> :: ModifyOptions (BaseAMGPC::Options & aO, const Flags & flags, string prefix)
{
  BaseStokesAMGPrecond::ModifyOptions(aO, flags, prefix);
} // NCStokesAMGPC::ModifyOptions


template<class TFACTORY>
shared_ptr<TopologicMesh> NCStokesAMGPC<TFACTORY> :: BuildAlgMesh (shared_ptr<BlockTM> &&top_mesh, FlatArray<FVDescriptor> fvd)
{
  static Timer t("NCStokesAMGPC::BuildAlgMesh");
  static Timer t1("NCStokesAMGPC::BuildAlgMesh - fill");

  RegionTimer rt(t);

  const auto & O = static_cast<Options&>(*options);


  auto &auxInfo = GetFacetAuxInfo();
  const auto & nc_uDofs = GetNCUDofs();

  /** ghost-vertices, volumes, loops **/
  shared_ptr<TMESH> alg_mesh = StokesAMGPrecond<TMESH>::BuildEmptyAlgMesh(std::move(top_mesh), fvd);

  auto edges = alg_mesh->template GetNodes<NT_EDGE>();

  auto ma = nc_fes->GetMeshAccess();

  auto comm = this->GetMA().GetCommunicator();

  /** compute facet-flow **/

  auto facet_flows = CalcFacetFlows(); // indexed by R-F-NR!
  auto &aed = *get<1>(alg_mesh->Data());
  auto edata = aed.Data();

  aed.SetParallelStatus(DISTRIBUTED); // just to be sure

  Array<int>  elnums(30);
  Array<int> delnums(30);

  RegionTimer rt1(t1);

  // iterate through "solid" edges where we have a local element
  for (auto ffnr : Range(auxInfo.GetNFacets_R())) {

    auto const  fnr  = auxInfo.R2A_Facet(ffnr);
    auto const  enr  = F2E(fnr);
    auto const &edge = edges[enr];
    auto nc_dps = nc_uDofs.GetDistantProcs(ffnr);

    double fac = 1.0;

    if ( nc_dps.Size() && (comm.Rank() > nc_dps[0]) ) {
      // Let the master of the NC-dof of the facet write the flow,
      // the master must know about both vertices of the edge so can orient
      // it correctly!
      fac = 0.0;
    }
    else {
      this->GetMA().GetFacetElements(fnr, elnums);
      // facet-flow is oriented outwards from lowest element!
      int start_elnr = ( (elnums.Size() > 1) && (elnums[0] > elnums[1]) ) ? elnums[1] : elnums[0];
      int vnr = this->EL2V(start_elnr);
      if (vnr == edge.v[0])
        { fac = 1.0; }
      else if (vnr == edge.v[1])
        { fac = -1.0; }
      else {
        throw Exception("CANNOT ORIENT elnr -> edge!");
      }
    }

    edata[enr].flow = fac * facet_flows[ffnr];

  }

  // weights
  switch(O.energy) {
    case(Options::TRIV_ENERGY):  { FillAlgMesh(*alg_mesh, [&](auto elnum, auto fnum, auto const &A) { return 1.0; }); break; }
    case(Options::ALG_ENERGY):   {
      FillAlgMesh(*alg_mesh, [&](auto elnum, auto fnum, auto const &A) {
        // maximum (by abs-of-trace) connection between facet "fnum" and any other facet in the same element
        auto di = auxInfo.A2R_Facet(fnum);
        auto facets0 = this->GetMA().GetElFacets(ElementId(VOL, elnum));
        double w0 = 0;
        for (auto fnumj : facets0) {
          if (fnumj != fnum) {
            int dj = auxInfo.A2R_Facet(fnumj);
            if (dj != -1) { // I think this cannot happen per definition (is a facet of an element!)
              // cout << " dj " << dj << endl;
              w0 = max2(w0, fabs(calc_trace(A(di, dj))));
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

  return alg_mesh;
} // NCStokesAMGPC::BuildAlgMesh


template<class TFACTORY>
template<class TLAM>
void NCStokesAMGPC<TFACTORY> :: FillAlgMesh(TMESH const &alg_mesh, TLAM weightLambda)
{
  auto &auxInfo = nc_fes->GetFacetAuxInfo();

  shared_ptr<BaseMatrix> wmat = (finest_weight_mat == nullptr) ? finest_mat : finest_weight_mat;
  shared_ptr<SparseMatrix<Mat<BS, BS, double>>> finest_spm;

  if (auto pmat = dynamic_pointer_cast<ParallelMatrix>( wmat ))
    { finest_spm = dynamic_pointer_cast<SparseMatrix<Mat<BS, BS, double>>>(pmat->GetMatrix()); }
  else
    { finest_spm = dynamic_pointer_cast<SparseMatrix<Mat<BS, BS, double>>>(wmat); }

  if (finest_spm == nullptr)
    { throw Exception("No valid matrix in NCStokesAMGPC::FillAlgMesh_ALG!"); }

  auto ma = nc_fes->GetMeshAccess();

  auto edges = alg_mesh.template GetNodes<NT_EDGE>();
  auto aed   = get<1>(alg_mesh.Data());
  auto edata = aed->Data();

  const auto & A = *finest_spm;

  Array<int> elnums;

  for (auto ffnr : Range(auxInfo.GetNFacets_R())) {

    auto const  fnr      = auxInfo.R2A_Facet(ffnr);
    auto const  enr      = F2E(fnr);
    auto const &edge     = edges[enr];

    this->GetMA().GetFacetElements(fnr, elnums);

    double wij = 0.0;
    double wji = 0.0;

    if (nc_fes->DefinedOn(ElementId(VOL, elnums[0])))
      { wij = weightLambda(elnums[0], fnr, A); }

    if ( (elnums.Size() == 2)  && nc_fes->DefinedOn(ElementId(VOL, elnums[1])) )
      { wji = weightLambda(elnums[1], fnr, A); }

    bool const flipped = EL2V(elnums[0]) == edge.v[1];

    if (flipped)
      { swap(wij, wji); }

    auto nc_dps = nc_uDofs.GetDistantProcs(ffnr);

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
Array<Vec<NCStokesAMGPC<TFACTORY>::BS, double>> NCStokesAMGPC<TFACTORY> :: CalcFacetFlows () const
{
  /**
   * Flow is oriented from lower element to higher element. ( = outside w.r.t smallest element )
   * So also oriented outside the domain.
   * So INCOSISTENT for MPI-BNDS ! (that is OK, I have to flip it later!)
   */
  static Timer t("NCStokesAMGPC::CalcFacetFlows");
  RegionTimer rt(t);

  LocalHeap lh(10 * 1024 * 1024, "Orrin_Oriscorin_Hiscorin_Sincorin_Alvin_Vladimir_Groolmoplong");
  // auto f2a_facet = nc_fes->GetFMapF2A();

  auto ma       = nc_fes->GetMeshAccess();
  auto &auxInfo = nc_fes->GetFacetAuxInfo();

  int curve_order = this->GetMA().GetCurveOrder(); // TODO: get this from mesh
  int ir_order    = 2 + this->GetMA().GetDimension() * curve_order;
  auto nv_cf = NormalVectorCF(this->GetMA().GetDimension());

  Array<Vec<BS, double>> flows(auxInfo.GetNFacets_R());

  Array<int> elnums;

  for (auto ffnr : Range(auxInfo.GetNFacets_R())) {

    HeapReset hr(lh);

    auto facet_nr = auxInfo.R2A_Facet(ffnr);

    /** TODO: this is hardcoded with the auxiliary facet elements -> convert to using the space's (VOL) els and not do it by hand anymore... **/
    NCH1FacetTrace<BS> auxfe (NodeId(NT_FACET, facet_nr), this->GetMA());
    this->GetMA().GetFacetElements(facet_nr, elnums); // defed or not really does not matter here!
    int start_elnr = ( (elnums.Size() > 1) && (elnums[0] > elnums[1]) ) ? elnums[1] : elnums[0];
    // GetDefinedFacetElements(facet_nr, elnums);
    ElementId ei(VOL, start_elnr);
    auto & trafo = this->GetMA().GetTrafo (ei, lh);
    auto facet_nrs = this->GetMA().GetElFacets(ei);
    int loc_facet_nr = facet_nrs.Pos(facet_nr);
    ELEMENT_TYPE et_vol = trafo.GetElementType();
    ELEMENT_TYPE et_facet = ElementTopology::GetFacetType (et_vol, loc_facet_nr);
    const IntegrationRule & ir_facet = SelectIntegrationRule (et_facet, ir_order); // reference facet
    Facet2ElementTrafo facet_2_el(et_vol, BND); // reference facet -> reference vol
    IntegrationRule & ir_vol = facet_2_el(loc_facet_nr, ir_facet, lh); // reference VOL
    ElementTransformation & eltrans = this->GetMA().GetTrafo (ei, lh);
    BaseMappedIntegrationRule & basemir = eltrans(ir_vol, lh);
    MappedIntegrationRule<BS,BS,double> & mir_vol(static_cast<MappedIntegrationRule<BS,BS,double>&>(basemir)); // mapped VOL
    mir_vol.ComputeNormalsAndMeasure(et_vol, loc_facet_nr);
    FlatVector<double> facet_flow(BS, lh); facet_flow = 0;
    FlatMatrix<double> auxval(BS, BS, lh);
    FlatVector<double> nvval(BS, lh);
    for (auto ip_nr : Range(mir_vol)) {
      auto mip = mir_vol[ip_nr];
      auxfe.CalcMappedShape(mip, auxval);
      nv_cf->Evaluate(mip, nvval);
      for (auto k : Range(BS))
        { facet_flow[k] += mip.GetWeight() * InnerProduct(auxval.Row(k), nvval); }
    }
    flows[ffnr] = facet_flow;
  }

  return flows;
} // NCStokesAMGPC::CalcFacetFlows


template<class TFACTORY>
BaseAMGFactory&
NCStokesAMGPC<TFACTORY>::
GetBaseFactory () const
{
  return GetFactory();
} // NCStokesAMGPC::GetBaseFactory


template<class TFACTORY>
SecondaryAMGSequenceFactory const&
NCStokesAMGPC<TFACTORY>::
GetSecondaryAMGSequenceFactory () const
{
  return GetFactory();
} // NCStokesAMGPC::GetSecondaryAMGSequenceFactory


template<class TFACTORY>
TFACTORY&
NCStokesAMGPC<TFACTORY>::
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
shared_ptr<BaseDOFMapStep> NCStokesAMGPC<TFACTORY> :: BuildEmbedding (BaseAMGFactory::AMGLevel & level)
{
  /**
   * We need a multi-embedding here:
   *   i) mesh edge-space      -> nc_fes
   *  ii) mesh potential-space -> nc_fes
   *        this is just a concatenation of the curl-matrix and the edge->nc_fes embedding from (i)
   *
   * Note: We need to take into account the difference between the facet-midpoint and vertex-midpoint position
   *       This does not matter for "BND"-vertices; Their pos is set s.t facet-MP is exactly the vertex-MP
   */

  auto const &O = static_cast<Options&>(*options);

  auto ma = nc_fes->GetMeshAccess();

  auto &auxInfo = nc_fes->GetFacetAuxInfo();

  shared_ptr<TopologicMesh> mesh = level.cap->mesh;
  const TMESH & M = static_cast<const TMESH&>(*mesh);
  M.CumulateData(); // !!

  /** nc -> mesh **/
  auto vdata = get<0>(M.Data())->Data();
  // auto & edge_sort = node_sort[NT_EDGE];
  // auto f2af = nc_fes->GetFMapF2A();
  // auto nc_pds = finest_mat->GetParallelDofs(); // dummy pardofs if sequential
  auto ncUDofs = MatToUniversalDofs(*finest_mat, DOF_SPACE::COLS);
  auto edges   = M.template GetNodes<NT_EDGE>();
  auto gvs     = M.GetGhostVerts();
  auto [dofed_edges, dofe2e, e2dofe] = M.GetDOFedEdges();

  Array<int> perow(ncUDofs.GetNDofLocal());
  perow = 1;

  auto mesh_emb = make_shared<typename TFACTORY::TSPM>(perow, dofe2e.Size());

  /** embedding mesh-range -> NC-space **/

  // H1 Qij is always identity...
  typename TFACTORY::ENERGY::TM Qij;
  SetIdentity(Qij);

  typename TFACTORY::ENERGY::TVD tvij, tvfacet;
  for (auto ffnr : Range(auxInfo.GetNFacets_R())) {
    // this is a "local" facet, so gg or sg -> the edge MUST be dofed
    auto fnr = auxInfo.R2A_Facet(ffnr); // FES dof-nr
    auto enr = this->F2E(fnr);
    auto dnr = e2dofe[enr];             // MESH dof-nr
    // auto enr = edge_sort[ffnr];
    if ( (enr == -1) || (dnr == -1) ) {
      cout << endl << " ffnr " << ffnr << " -> fnr " << fnr << " , enr " << enr << " -> dof " << e2dofe[enr] << endl;
      const auto & edge = edges[enr];
      cout << " edge " << edge << endl;
      cout << " edata " << get<0>(M.Data())->Data()[enr] << endl;
      cout << " vgs " << (gvs->Test(edge.v[0])?1:0) << " " << (gvs->Test(edge.v[1])?1:0) << endl;
      cout << " vdatas " << vdata[edge.v[0]] << " " << vdata[edge.v[1]] << endl;
      Array<int> facet_els;
      nc_fes->GetMeshAccess()->GetFacetElements(fnr, facet_els);
      cout << " all      facet_els : "; prow(facet_els); cout << endl;
      this->GetDefinedFacetElements(fnr, facet_els); // E-NR, not DE-NR
      cout << " defined facet_els : "; prow(facet_els); cout << endl;
      if (nc_fes->GetParallelDofs())
      cout << " nc_dps : "; prow(nc_fes->GetParallelDofs()->GetDistantProcs(ffnr)); cout << endl;
      cout << " ma facet dps "; prow(nc_fes->GetMeshAccess()->GetDistantProcs(NodeId(NT_FACET, fnr))); cout << endl;
    }
    const auto & edge = edges[enr];
    if constexpr(is_same<typename TFACTORY::ENERGY::TVD::TVD, double>::value == 0) {
      GetNodePos<TFACTORY::DIM>(NodeId(FACET_NT(TFACTORY::DIM), fnr), this->GetMA(), tvfacet.vd.pos); // cheating ... WHY is it cheating, seems reasonable ??
      tvij = TFACTORY::ENERGY::CalcMPData(vdata[edge.v[0]], vdata[edge.v[1]]);
      TFACTORY::ENERGY::ModQHh(tvfacet, tvij, Qij);
    }
    (*mesh_emb)(ffnr, dnr) = Qij;
  }

  // if (M.GetEQCHierarchy()->GetCommunicator().Rank() == 35)
    // { cout << "mesh_emb:" << endl; print_tm_spmat(cout, *mesh_emb); cout << endl << endl << endl; }

  auto mesh_uDofs = GetFactory().BuildUDofs(*level.cap);
  auto emb_dms    = make_shared<ProlMap<typename TFACTORY::ENERGY::TM>>(mesh_emb, nc_uDofs, mesh_uDofs);

  /** embedding mesh-potential -> NC-space **/

  auto lcc = static_pointer_cast<typename TFACTORY::StokesLevelCapsule>(level.cap);
  GetFactory().BuildCurlMat(*lcc);
  GetFactory().BuildPotUDofs(*lcc);
  auto emb_prol = emb_dms->GetProl();

  // TODO: AAARGGS I HATE IT
  auto &cmat = static_cast<typename TFACTORY::TCM_TM &>(*lcc->curl_mat);

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

  auto cep = MatMultAB(*emb_prol, cmat);
  auto emb_pot = make_shared<ProlMap<StripTM<TFACTORY::BS, 1>>>(cep, nc_uDofs, lcc->pot_uDofs);

  // if (M.GetEQCHierarchy()->GetCommunicator().Rank() == 35) {
  //   cout << endl << endl << " final emb mat " << endl; print_tm_spmat(cout << endl, *cep); cout << endl;
  //   cout << endl << endl << " final mat / emb dims " << finest_mat->Height() << " x " << finest_mat->Width() << " times " <<
  // 	cep->Height() << " x " << cep->Width() << endl;
  // }

  /** combined **/
  Array<shared_ptr<BaseDOFMapStep>> steps( { emb_dms, emb_pot } );
  auto multi_emb = make_shared<MultiDofMapStep>(steps);

  if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG)
  {
    auto const rk = multi_emb->GetUDofs().GetCommunicator().Rank();

    std::ofstream of("embedding_rk_" + std::to_string(rk) + ".out");

    of << *multi_emb << endl;
  }

  return multi_emb;
} // NCStokesAMGPC::BuildEmbedding


template<class TFACTORY>
void NCStokesAMGPC<TFACTORY> :: InitFinestLevel (BaseAMGFactory::AMGLevel & afinest_level)
{
  auto & O(static_cast<Options&>(*options));
  if (O.force_ass_flmat)
    { throw Exception("force_ass_flmat for stokes probably not correct yet"); }

  BaseAMGPC::InitFinestLevel(afinest_level);

  auto stokes_cap = static_pointer_cast<typename TFACTORY::StokesLevelCapsule>(afinest_level.cap);

  auto &fmesh = static_cast<TMESH&>(*stokes_cap->mesh);
  auto floops = fmesh.GetLoops();


  /** Dirichlet Conditions **/

  auto [free_verts, free_edges, free_psn] = this->BuildFreeNodes(fmesh, floops, fmesh.GetLoopUDofs());

  stokes_cap->pot_freedofs = free_psn;   // potential space smoother freedofs
  stokes_cap->free_nodes   = free_verts; // coarsening "freedofs"
  fmesh.SetFreeNodes(free_edges);        // for prolongation

} // NCStokesAMGPC::InitFinestLevel


template<class TFACTORY>
void NCStokesAMGPC<TFACTORY> :: FinalizeLevel (shared_ptr<BaseMatrix> mat)
{
  nc_uDofs = MatToUniversalDofs(*mat, DOF_SPACE::COLS);

  StokesAMGPrecond<TMESH>::FinalizeLevel(mat);
} // NCStokesAMGPC::FinalizeLevel


template<class TFACTORY>
FacetAuxiliaryInformation const & NCStokesAMGPC<TFACTORY> :: GetFacetAuxInfo () const
{
  return nc_fes->GetFacetAuxInfo();
} // NCStokesAMGPC::SetFacetAuxInfo

// template<class TFACTORY>
// std::tuple<Array<shared_ptr<BaseAMGFactory::AMGLevel>>,
//            Array<shared_ptr<BaseDOFMapStep>>,
//            shared_ptr<DOFMap>>
// NCStokesAMGPC<TFACTORY>::
// CreateSecondaryAMGSequence(FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>>        amgLevels,
//                           DOFMap                                          const &dOFMap) const
// {
//   throw Exception("NCStokesAMGPC::CreateSecondaryAMGSequence");

//   Array<shared_ptr<BaseAMGFactory::AMGLevel>> secLevels;
//   Array<shared_ptr<BaseDOFMapStep>> secEmbs;
//   shared_ptr<DOFMap> secDOFMap;

//   return std::make_tuple(secLevels, secEmbs, secDOFMap);
// }



  // template<class TFACTORY>
  // void NCStokesAMGPC<TFACTORY> :: SetLoops (shared_ptr<typename NCStokesAMGPC<TFACTORY>::TMESH> mesh)
  // {
  //   /** Set the topological loops for Hiptmair Smoother **/
  //   auto loops = CalcFacetLoops(mesh);
  //   auto free_facets = nc_fes->GetFreeDofs();
  //   int cntrl = 0;
  //   for (auto k : Range(loops)) {
  //     bool takeloop = true;
  //     for (auto j : loops[k]) {
  // 	auto fnr = abs(j) - 1;
  // 	if (!free_facets->Test(fnr))
  // 	  { takeloop = false; /** cout << " discard loop " << k << endl; **/ }
  //     }
  //     if (takeloop)
  // 	{ cntrl++; }
  //   }
  //   auto a2f_facet = nc_fes->GetFMapA2F();
  //   FlatArray<int> vsort = node_sort[0];
  //   FlatArray<int> fsort = node_sort[1];
  //   auto edges = mesh->template GetNodes<NT_EDGE>();
  //   Array<int> elnums;
  //   TableCreator<int> crl(cntrl);
  //   for (; !crl.Done(); crl++) {
  //     int c = 0;
  //     for (auto loop_nr : Range(loops)) {
  // 	auto loop = loops[loop_nr];
  // 	bool takeloop = true;
  // 	for(auto j : Range(loop)) {
  // 	  int fnr = abs(loop[j])-1;
  // 	  int enr = fsort[a2f_facet[fnr]];
  // 	  const auto & edge = edges[enr];
  // 	  double fac = 1;
  // 	  this->GetMA().GetFacetElements(fnr, elnums);
  // 	  if (vsort[elnums[0]] == edge.v[0])
  // 	    { fac = 1.0; }
  // 	  else if (vsort[elnums[0]] == edge.v[1])
  // 	    { fac = -1; }
  // 	  else
  // 	    { fac = 1; } // doesnt matter - should remove these loops anyways (?)
  // 	  // if (enr != -1) { // let's say this cannot happen ??
  // 	  // actually, I think we should throw out loops that touch the Dirichlet BND
  // 	  loop[j] = (loop[j] > 0) ? fac * (1 + enr) : -fac * (1 + enr);
  // 	  if (!free_facets->Test(fnr))
  // 	    { takeloop = false; /** cout << " discard loop " << loop_nr << endl; **/ break; }
  // 	  // }
  // 	}
  // 	if (takeloop)
  // 	  { crl.Add(c++, loop); }
  //     }
  //   }
  //   auto edata = get<1>(mesh->Data())->Data();
  //   // cout << " edges/flows: " << endl;
  //   // for (auto k : Range(edges.Size()))
  //   // { cout << edges[k] << ", flow " << edata[k].flow << endl; }
  //   auto mod_loops = crl.MoveTable();
  //   // cout << " modded loops: " << endl << mod_loops << endl;
  //   // mesh->SetLoops(std::move(loops));
  //   mesh->SetLoops(std::move(mod_loops));
  // } // NCStokesAMGPC::SetLoops


  /** END NCStokesAMGPC **/

} // namespace amg


#endif // FILE_NC_STOKES_PC_IMPL_HPP
