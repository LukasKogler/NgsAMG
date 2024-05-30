#ifndef FILE_AMG_FACTORY_STOKES_NC_IMPL_HPP
#define FILE_AMG_FACTORY_STOKES_NC_IMPL_HPP

#include <utils_io.hpp>
#include <alg_mesh_nodes.hpp>

#include "stokes_map.hpp"
#include "stokes_factory.hpp"

#include "stokes_map_impl.hpp"

#include "stokes_factory_impl.hpp"

#include "nc_stokes_factory.hpp"
namespace amg
{


/** NCStokesAMGFactory **/

template<class TMESH, class ENERGY>
NCStokesAMGFactory<TMESH, ENERGY> :: NCStokesAMGFactory (shared_ptr<NCStokesAMGFactory<TMESH, ENERGY>::Options> _opts)
  : BASE_CLASS(_opts)
{
  ;
} // NCStokesAMGFactory(..)


template<class TMESH, class ENERGY>
shared_ptr<typename NCStokesAMGFactory<TMESH, ENERGY>::TSPM>
NCStokesAMGFactory<TMESH, ENERGY> ::
BuildPrimarySpaceProlongation (BaseAMGFactory::LevelCapsule const &fcap,
                               BaseAMGFactory::LevelCapsule       &ccap,
                               StokesCoarseMap<TMESH>       const &cmap,
                               TMESH                        const &fmesh,
                               TMESH                              &cmesh,
                               FlatArray<int>                      vmap,
                               FlatArray<int>                      emap,
                               FlatTable<int>                      v_aggs,
                               TSPM_TM                      const *pA)
{
  auto & O = static_cast<Options&>(*options);

  /**
    *   Stokes prolongation works like this:
    *        Step  I) Find a prolongation on agglomerate facets
    *  Step II) Extend prolongation to agglomerate interiors:
    *              Minimize energy. (I) as BC. Additional constraint:
    *          \int_a div(u) = |a|/|A| \int_A div(U)    (note: we maintain constant divergence)
    *
    *   "Piecewise" Stokes Prolongation.
    *       Step I) Normal PW Prolongation as in standard AMG
    *
    *   "Smoothed" Stokes Prol takes PW Stokes prol and then:
    *       Step  I) Take PW prol, then smooth on facets between agglomerates. Additional constraint:
    *           Maintain flow through facet.
    */
  static Timer t("NCStokesAMGFactory::BuildPWProl_impl");
  RegionTimer rt(t);

  static constexpr int BS = mat_traits<TM>::HEIGHT;

  /** fine mesh **/
  const auto & FM(fmesh);

  FM.CumulateData();
  const auto & fecon(*FM.GetEdgeCM());
  auto fvd = get<0>(FM.Data())->Data();
  auto fed = get<1>(FM.Data())->Data();
  size_t FNV = FM.template GetNN<NT_VERTEX>(), FNE = FM.template GetNN<NT_EDGE>();
  auto fgv = FM.GetGhostVerts();
  auto free_fes = FM.GetFreeNodes();
  const auto & spA = *pA;

  /** coarse mesh **/
  const auto & CM(cmesh);

  CM.CumulateData();
  const auto & cecon(*CM.GetEdgeCM());
  auto cvd = get<0>(CM.Data())->Data();
  auto ced = get<1>(CM.Data())->Data();
  auto cgv = FM.GetGhostVerts();
  size_t CNV = CM.template GetNN<NT_VERTEX>(), CNE = CM.template GetNN<NT_EDGE>();

  const auto & eqc_h = *FM.GetEQCHierarchy();

  // const bool bdoco = eqc_h.GetCommunicator().Size() == 4;
  const bool bdoco = fcap.baselevel == 1;

  /** DOF <-> edge **/
  // auto [c_dofed_edges, c_dof2e, c_e2dof] = CM.GetDOFedEdges();
  // auto [f_dofed_edges, f_dof2e, f_e2dof] = FM.GetDOFedEdges();

  auto [c_dofed_edges_SB, c_dof2e_SB, c_e2dof_SB] = CM.GetDOFedEdges();
  auto &c_dofed_edges = c_dofed_edges_SB;
  auto &c_dof2e = c_dof2e_SB;
  auto &c_e2dof = c_e2dof_SB;

  auto [f_dofed_edges_SB, f_dof2e_SB, f_e2dof_SB] = FM.GetDOFedEdges();
  auto &f_dofed_edges = f_dofed_edges_SB;
  auto &f_dof2e = f_dof2e_SB;
  auto &f_e2dof = f_e2dof_SB;

  size_t FND = f_dof2e.Size(), CND = c_dof2e.Size();

  // cout << "CRS VMAP " << vmap.Size() << endl;
  // for (auto k : Range(vmap))
  //   cout << "FV" << k << " -> CV" << vmap[k] << endl;
  // cout << endl << endl;

  if (free_fes != nullptr) {
    auto fedges = FM.template GetNodes<NT_EDGE>();
    auto cedges = CM.template GetNodes<NT_EDGE>();
    for (auto fenr : Range(free_fes->Size())) {
      if (!free_fes->Test(fenr)) {
        auto cenr = emap[fenr];
        if (cenr != -1) { // reminder: also some "interior" problem edges!
          const auto & fedge = fedges[fenr];
          const auto & cedge = cedges[cenr];
          cout << " DRIFENR " << fenr << " -> " << cenr << endl;
          cout << " fine/crs dof-nrs = " << f_e2dof[fenr] << " -> " << c_e2dof[cenr] << endl;
          cout << " fine edge = " << fedges[fenr] << " -> coarse edge = " << cedges[cenr] << endl;
          cout << " fine ed " << fed[fenr] << " -> coarse ed " << ced[cenr] << endl;
          cout << " fine vgs " << (fgv->Test(fedge.v[0])?1:0) << " " << (fgv->Test(fedge.v[1])?1:0) << endl;
          cout << " fine vds " << fvd[fedge.v[0]] << " | " << fvd[fedge.v[1]] << endl;
          cout << " crs  vgs " << (cgv->Test(cedge.v[0])?1:0) << " " << (cgv->Test(cedge.v[1])?1:0) << endl;
          cout << " crs  vds " << cvd[cedge.v[0]] << " | " << cvd[cedge.v[1]] << endl;
          cout << "vmaps " << vmap[fedge.v[0]] << " " << vmap[fedge.v[1]] << endl;
        }
      }
    }
  }

  // {
  //   auto fedges = FM.template GetNodes<NT_EDGE>();
  //   auto cedges = CM.template GetNodes<NT_EDGE>();
  //   auto ife = [&](auto & edge) {
  //     if ( eqc_h.GetCommunicator().Rank() == 33)
  //       { return ( (edge.v[0] == 432294) || (edge.v[1] == 432294) ); }
  //     if ( eqc_h.GetCommunicator().Rank() == 35)
  //       { return ( (edge.v[0] == 425201) || (edge.v[1] == 425201) ); }
  //     return false;
  //   };
  //   for (auto fenr : Range(fedges)) {
  //     if (ife(fedges[fenr])) {
  //       const auto & fedge = fedges[fenr];
  //       cout << "INTERESTING FEDGE " << fedges[fenr] << endl;
  //       cout << " fine/crs dof-nrs = " << f_e2dof[fenr] << endl;
  //       cout << " fine edge = " << fedges[fenr] << endl;
  //       cout << " fine ed " << fed[fenr] << endl;
  //       cout << " fine vgs " << (fgv->Test(fedge.v[0])?1:0) << " " << (fgv->Test(fedge.v[1])?1:0) << endl;
  //       cout << " fine vds " << fvd[fedge.v[0]] << " | " << fvd[fedge.v[1]] << endl;
  //       cout << "vmaps " << vmap[fedge.v[0]] << " " << vmap[fedge.v[1]] << endl;
  //       int cv0 = vmap[fedge.v[0]], cv1 = vmap[fedge.v[1]];
  //       if ( cv0 != -1 ) {
  //         cout << "CV0 gh " << (cgv->Test(cv0)?1:0) << ", vd = " << cvd[cv0] << endl;
  //       }
  //       if ( cv1 != -1 ) {
  //         cout << "CV1 gh " << (cgv->Test(cv1)?1:0) << ", vd = " << cvd[cv1] << endl;
  //       }
  //       auto cenr = emap[fedge.id];
  //       if ( cenr != -1) {
  //         const auto & cedge = fedges[cenr];
  //         cout << "CEDGE " << cedge << endl;
  //         cout << " C DNR -> " << c_e2dof[cenr] << endl;
  //         cout << " CED " << ced[cenr] << endl;
  //         cout << " crs  vgs " << (cgv->Test(cedge.v[0])?1:0) << " " << (cgv->Test(cedge.v[1])?1:0) << endl;
  //         cout << " crs  vds " << cvd[cedge.v[0]] << " | " << cvd[cedge.v[1]] << endl;
  //       }
  //     }
  //   }
  // }

  // if (bdoco) {
  //   cout << " BuildPWProl_impl  FMESH = " << endl << *fmesh << endl;
  //   cout << "fecon: " << endl << fecon << endl;
  //   cout << endl << " F dof -> edge "; prow2(f_dof2e); cout << endl;
  //   cout << " F edge -> dof "; prow2(f_e2dof); cout << endl;
  //   cout << endl << "vmap: "; prow2(vmap); cout << endl << endl;
  //   cout << endl << "emap: "; prow2(emap); cout << endl << endl;
  //   cout << endl << " BuildPWProl_impl  CMESH = " << endl << *cmesh << endl;
  //   cout << endl << "cecon: " << endl << cecon << endl;
  //   cout << " C dof -> edge "; prow2(c_dof2e); cout << endl;
  //   cout << " C edge -> dof "; prow2(c_e2dof); cout << endl;
  // }

  /** prol dims **/
  size_t H = FND, W = CND;

  /** Count entries **/
  Array<int> perow(H); perow = 0;
  auto fedges = FM.template GetNodes<NT_EDGE>();
  for (auto fdnr : Range(H)) {
    auto fenr = f_dof2e[fdnr];
    auto & fedge = fedges[fenr];
    if ( free_fes && (!free_fes->Test(fenr)) ) // dirichlet // TODO: test for D or E nr?? (I think E)
    { perow[fdnr] = 0; }
    else {
      auto cenr = emap[fenr];
      if (cenr != -1) // edge connects two agglomerates, at least one must be dofed
        { perow[fdnr] = 1; }
      else { // verts in an agg are all S or all G, edge is sg or ss -> verts must be s, edge must be ss
        int cv0 = vmap[fedge.v[0]], cv1 = vmap[fedge.v[1]];
        if (cv0 == cv1) {
          if (cv0 == -1) { // two fine verts that map to -1 ... should happen basically never
            // should only be possible due to l2 weight vertex collapse
            // as I am not doing that ATM, keep an exception here for now
            cout << "WEIRD CASE A!! for fedge " << fedge << ", FENR = " << fenr << " of " << fedges.Size() << " " << FM.template GetNN<NT_EDGE>() << endl;
            cout << " cenr " << cenr << ", cvs " << cv0 << " " << cv1 << endl;
            perow[fdnr] = 0;
            throw Exception("Weird case A!");
          }
          else // edge is interior to an agglomerate - alloc entries for all facets of the agglomerate (must all be ss/sg)
            { perow[fdnr] = cecon.GetRowIndices(cv0).Size(); }
        }
        else { // edge between a Dirichlet BND vertex and an interior one
          perow[fdnr] = 0;
          // throw Exception("Weird case B!");
        }
      }
    }
  }

  /** Allocate Matrix  **/
  auto prol = make_shared<TSPM> (perow, W);
  auto & P(*prol);
  const auto & const_P(*prol);
  P.AsVector() = -42;

  /** Fill facets **/
  typename ENERGY::TVD tvf, tvc;
  for (auto fdnr : Range(H)) {
    auto fenr = f_dof2e[fdnr];
    auto & fedge = fedges[fenr];
    auto cenr = emap[fenr];
    if (cenr != -1) { // fine must be ss/sg -> coarse must be too
      int fv0 = fedge.v[0], fv1 = fedge.v[1];
      tvf = ENERGY::CalcMPData(fvd[fv0], fvd[fv1]);
      int cv0 = vmap[fv0], cv1 = vmap[fv1];
      tvc = ENERGY::CalcMPData(cvd[cv0], cvd[cv1]);
      P.GetRowIndices(fdnr)[0] = c_e2dof[cenr];
      ENERGY::CalcQHh(tvc, tvf, P.GetRowValues(fdnr)[0]);
    }
  }

  // cout << "stage 1 pwp: " << endl;
  // print_tm_spmat(cout << endl, P);


  /** Extend the prolongation - Solve:
      |A_vv B_v^T|  |u_v|     |-A_vf      0     |  |P_f|
      |  ------  |  | - |  =  |  -------------  |  | - | |u_c|
      |B_v       |  |lam|     |-B_vf  d(|v|/|A|)|  |B_A|
    **/
  LocalHeap lh(30 * 1024 * 1024, "Jerry");
  for (auto agg_nr : Range(v_aggs)) {

    HeapReset hr(lh);

    auto agg_vs = v_aggs[agg_nr];

    // ghost-agg: has no interior DOFs
    if ( (fgv != nullptr) && fgv->Test(agg_vs[0]))
      { continue; }

    // for single verts there are no interior edges
    if (agg_vs.Size() <= 1)
      { continue; }

    // now must be a solid-agg -> all interior edges are ss
    auto cv = vmap[agg_vs[0]];

    // const bool doco = bdoco && ( (agg_vs.Contains(84) || agg_vs.Contains(96) || agg_vs.Contains(1246) || agg_vs.Contains(1245) || (cv == 146) ) );
    // const bool doco = (cv == 486);
    // const bool doco = bdoco;
    // if (doco) {
    //   cout << endl << "FILL AGG FOR CV " << cv << "/" << v_aggs.Size() << endl;
    //   cout << endl << "FILL AGG FOR CV " << cv << "/" << v_aggs.Size() << endl;
    //   cout << "fill agg " << agg_nr << ", agg_vs: "; prow(agg_vs); cout << endl;
    //   cout << "v vols: " << endl;
    //   for (auto v : agg_vs)
    //     { cout << fvd[v].vol << " "; }
    //   cout << endl;
    // }
    auto cneibs = cecon.GetRowIndices(cv);
    auto cfacets = cecon.GetRowValues(cv);
    int nfv = agg_vs.Size(), ncv = 1;     // # fine/coarse elements
    /** count fine facets **/
    int ncf = cfacets.Size();             // # coarse facets
    int nff = 0; int nffi = 0, nfff = 0;  // # fine facets (int/facet)
    auto it_f_facets = [&](auto lam) {
      for (auto k : Range(agg_vs)) {
        auto vk = agg_vs[k];
        // if (doco) cout << k << ", vk " << vk << endl;
            auto vk_neibs = fecon.GetRowIndices(vk);
        // if (doco) { cout << "neibs "; prow(vk_neibs); cout << endl; }
            auto vk_fs = fecon.GetRowValues(vk);
        for (auto j : Range(vk_neibs)) {
          auto vj = vk_neibs[j];
          auto cvj = vmap[vj];
          // if (doco) cout << j << " vj " << vj << " cvj " << cvj << endl;
          if (cvj == cv) { // neib in same agg - interior facet!
            auto kj = find_in_sorted_array(vj, agg_vs);
            if (vj > vk) { // do not count interior facets twice!
              lam(vk, k, vj, kj, int(vk_fs[j]));
            }
          }
          else // neib in different agg (or dirichlet)
            { lam(vk, k, vj, -1, int(vk_fs[j])); }
        }
      }
    };
    double avg_flow = 0.0, cnt_flow = 0.0;
    it_f_facets([&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
        nff++;
        if (kj == -1) // ex-facet
          { nfff++; }
        else
          { nffi++; }
        avg_flow += L2Norm(fed[eid].flow); cnt_flow += 1.0;
      });
    avg_flow /= cnt_flow;
    // if (doco) cout << "nff/f/i " << nff << " " << nfff << " " << nffi << endl << endl;
    /** All vertices (in this connected component) are in one agglomerate - nothing to do!
        Sometimes happens on coarsest level. **/
    if (nfff == 0)
      { continue; }

    /** I think this can only happen for my special test case with virtual vertex aggs ! **/
    if (nffi == 0)
      { continue; }
    /** fine facet arrays **/
    FlatArray<int> index(nff, lh), buffer(nff, lh);
    FlatArray<int> ffacets(nff, lh), ffiinds(nffi, lh), fffinds(nfff, lh);
    nff = nffi = nfff = 0;
    it_f_facets([&](int vi, int ki, int vj, int kj, int eid) LAMBDA_INLINE {
        index[nff] = nff;
        if (kj == -1)
          { fffinds[nfff++] = nff; }
        else
          { ffiinds[nffi++] = nff; }
        ffacets[nff++] = eid;
      });
    // if (doco) { cout << "unsorted ffacets " << nff << ": "; prow(ffacets, cout << endl); cout << endl; }
    // if (doco) { cout << "unsorted ffiinds " << nffi << ": "; prow(ffiinds, cout << endl); cout << endl; }
    // if (doco) { cout << "unsorted fffinds " << nfff << ": "; prow(fffinds, cout << endl); cout << endl; }
    QuickSortI(ffacets, index);
    for (auto k : Range(nffi))
      { ffiinds[k] = index.Pos(ffiinds[k]); }
    for (auto k : Range(nfff)) // TODO...
      { fffinds[k] = index.Pos(fffinds[k]); }
    // if (doco) { cout << "index " << nff << ": "; prow2(index, cout << endl); cout << endl; }
    ApplyPermutation(ffacets, index);
    QuickSort(fffinds);
    QuickSort(ffiinds);
    // if (doco) { cout << "ffacets " << nff << ": "; prow(ffacets, cout << endl); cout << endl; }
    // if (doco) { cout << "ffiinds " << nffi << ": "; prow(ffiinds, cout << endl); cout << endl; }
    // if (doco) { cout << "fffinds " << nfff << ": "; prow(fffinds, cout << endl); cout << endl; }

    auto it_f_edges = [&](auto lam) {
      for (auto kvi : Range(agg_vs)) {
        auto vi = agg_vs[kvi];
        auto vi_neibs = fecon.GetRowIndices(vi);
        auto vi_fs = fecon.GetRowValues(vi);
        const int nneibs = vi_fs.Size();
        for (auto lk : Range(nneibs)) {
          auto vk = vi_neibs[lk];
          auto fik = int(vi_fs[lk]);
          auto kfik = find_in_sorted_array(fik, ffacets);
          for (auto lj : Range(lk)) {
            auto vj = vi_neibs[lj];
            auto fij = int(vi_fs[lj]);
            auto kfij = find_in_sorted_array(fij, ffacets);
            lam(vi, kvi, vj, vk, fij, kfij, fik, kfik);
          }
        }
      }
    };

    int HA = nff * BS, HB = 0;

    const double eps = 1e-12;

    auto it_real_vs = [&](auto lam) {
      int rkvi = 0;
      for (auto kvi : Range(agg_vs)) {
        auto vi = agg_vs[kvi];
        auto v_vol = fvd[vi].vol;
        lam(v_vol, vi, kvi, rkvi);
        if (v_vol > 0.0)
          { rkvi++; }
      }
    };
    double cvol = 0.0; bool has_outflow = false;
    it_real_vs([&](auto vol, auto vi, auto kvi, auto rkvi) {
        if (vol < 0.0)
          { has_outflow = true; }
        else if (vol > 0.0)
          { cvol += vol; HB++; }
      });
    // if (doco) cout << " CVOL I " << cvol << " VS " << cvd[cv].vol << endl;
    /** If there is more than one element and no outflow facet, we need to lock constant pressure.
        If we have only one "real" element, there should always be an outflow, I believe. **/
    bool lock_const_pressure = (!has_outflow) && (HB > 1);
    if ( lock_const_pressure )
      { HB++; }

    int HM = HA + HB;

    // if (doco) cout << "LCP = " << lock_const_pressure << endl;
    // if (doco) cout << "H A/B/M = " << HA << " / " << HB << " / " << HM << endl;

    FlatMatrix<double> M(HM, HM, lh); M = 0;
    auto A = M.Rows(0, HA).Cols(0, HA);

    /** A block **/
    // FlatMatrix<double> A(nff * BS, nff * BS, lh); A = 0;
    auto a = M.Rows(0, HA).Cols(0, HA);
    FlatMatrix<TM> eblock(2,2,lh);

    it_f_edges([&](auto vi, auto kvi, auto vj, auto vk, auto fij, auto kfij, auto fik, auto kfik) {
      /** Only an energy-contrib if neither vertex is Dirichlet or grounded. Otherwise kernel vectors are disturbed. **/
      if ( (vmap[vj] != -1) && (vmap[vk] != -1) ) {
        ENERGY::CalcRMBlock(eblock, fvd[vi], fvd[vj], fvd[vk], fed[fij], (vi > vj), fed[fik], (vi > vk));
        // if (doco) {
        //   cout << "block verts " << vi << "-" << vj << "-" << vk << endl;
        //   cout << "block faces " << fij << "-" << fik << endl;
        //   cout << "v-data " << fvd[vi] << " " << fvd[vj] << endl;
        //   cout << "e-data " << fed[fij] << " " << fed[fik] << endl;
        //   cout << "block loc faces " << kfij << "-" << kfik << endl;
        //   cout << "block:" << endl; print_tm_mat(cout, eblock); cout << endl;
        // }
        Iterate<2>([&](auto i) {
          int osi = BS * (i.value == 0 ? kfij : kfik);
          Iterate<2>([&](auto j) {
            int osj = BS * (j.value == 0 ? kfij : kfik);
            A.Rows(osi, osi + BS).Cols(osj, osj + BS) += eblock(i.value, j.value);
          });
        });
      }
    });

    // A = 0.0;
    // it_f_edges([&](auto vi, auto kvi, auto vj, auto vk, auto fij, auto kfij, auto fik, auto kfik) {
    //     /** Only an energy-contrib if neither vertex is Dirichlet or grounded. Otherwise kernel vectors are disturbed. **/
    //     if ( (vmap[vj] != -1) && (vmap[vk] != -1) ) {
    //       int osi = BS * kfij, osj = BS * kfik;
    //       A.Rows(osi, osi+BS).Cols(osi, osi+BS) = spA(f_e2dof[fij], f_e2dof[fij]);
    //       A.Rows(osi, osi+BS).Cols(osj, osj+BS) += spA(f_e2dof[fij], f_e2dof[fik]);
    //       A.Rows(osj, osj+BS).Cols(osi, osi+BS) += spA(f_e2dof[fik], f_e2dof[fij]);
    //       A.Rows(osj, osj+BS).Cols(osj, osj+BS) = spA(f_e2dof[fik], f_e2dof[fik]);
    //     }
    //   });

    // if (doco) cout << "A block: " << endl << A << endl;

    // scale A and B block to same order as A block for numerical
    // stability
    double avg_da = 0.0;
    for (auto k : Range(A.Height()))
      { avg_da += A(k,k); }
    avg_da /= A.Height();

    /** (fine) B blocks **/
    auto Bf = M.Rows(HA, HA+HB).Cols(0, HA);
    auto BfT = M.Rows(0, HA).Cols(HA, HA+HB);
    double bfsum = 0;
    it_real_vs([&](auto vol, auto vi, auto kvi, auto rkvi) {
      if ( vol <= 0.0 )
        { return; }
      auto BfRow = Bf.Row(rkvi);
      auto vi_neibs = fecon.GetRowIndices(vi);
      auto vi_fs = fecon.GetRowValues(vi);
      // if (doco) cout << "calc b for " << kvi << ", vi " << vi << ", rkvi " << rkvi << endl;
      // if (doco){  cout << "neibs "; prow(vi_neibs); cout << endl; }
        // if (doco) { cout << "edgs "; prow(vi_fs); cout << endl; }
      const double fac = 1.0/fvd[vi].vol;
      for (auto j : Range(vi_fs)) {
        auto vj = vi_neibs[j];
        auto fij = int(vi_fs[j]);
        auto kfij = find_in_sorted_array(fij, ffacets);
        auto & fijd = fed[fij];
        int col = BS * kfij;
        // if (doco) cout << "j " << j << ", vj " << vj << ", fij " << fij << ", kfij " << kfij << ", col " << col << ", fac " << fac << endl;
        // if (doco) cout << "vol volinv: " << fvd[vi].vol << " " << fac << endl;
        // if (doco) cout << "flow: " << fijd.flow << endl;
        for (auto l : Range(BS)) {
          BfRow(col++) = ( (vi < vj) ? 1.0 : -1.0) * fijd.flow(l);
          // BfRow(col++) = ( (vi < vj) ? fac : -fac) * fijd.flow(l);
          bfsum += fac * abs(fijd.flow(l));
        }
      }
      });
    // if (doco) cout << " avgs " << avg_flow << " " << avg_da << endl;

    // avg_da = 1.0; avg_flow = 1.0;

    double cvinv0 = 1.0 / cvol;

    Bf *= avg_da / avg_flow * cvinv0;
    BfT = Trans(Bf);

    /** (coarse) B **/
    FlatArray<double> bcbase(BS * ncf, lh);
    FlatMatrix<double> Bc(HB, BS * ncf, lh);
    auto cv_neibs = cecon.GetRowIndices(cv);
    auto cv_fs = cecon.GetRowValues(cv);
    int bccol = 0;
    for (auto j : Range(cv_fs)) {
      auto cvj = cv_neibs[j];
      auto fij = int(cv_fs[j]);
      auto & fijd = ced[fij];
      // if (doco) cout << " c vol " << cvd[cv].vol << " " << 1.0/cvd[cv].vol << endl;
      // if (doco) cout << " cvj " << cvj << " cfij " << fij << endl;
      // if (doco) cout << " c flow " << fijd.flow << endl;
      for (auto l : Range(BS))
        { bcbase[bccol++] = avg_da/avg_flow * ( (cv < cvj) ? 1.0 : -1.0) * fijd.flow(l); }
    }

    // if (doco) {cout << " bcbase " << endl; prow(bcbase); cout << endl; }

    /** If we have an outflow, we force 0 divergence, otherwise only constant divergence **/
    // do not use cvd[cv].vol here as it can have empty volume ("immersed" verts with only one neib)
    // const double cvinv = (has_outflow) ? 0.0 : 1.0 / cvol;
    double cvinv = (has_outflow) ? 0.0 : 1.0 / cvol;
    if (lock_const_pressure)
      { cvinv = 0.0; }
    Bc = 0; // last row of Bc is kept 0 for pressure avg. constraint
    it_real_vs([&](auto vol, auto vi, auto kvi, auto rkvi) {
        if ( vol <= 0.0 )
          { return; }
        auto bcrow = Bc.Row(rkvi);
        for (auto col : Range(bcbase))
          { bcrow(col) = cvinv * fvd[vi].vol * bcbase[col]; }
        // for (auto col : Range(bcbase))
          // { bcrow(col) = cvinv * bcbase[col]; }
      });

    /** RHS **/
    int Hi = BS * nffi, Hf = BS * nfff, Hc = BS * ncf;
    FlatArray<int> colsi(Hi + HB, lh), colsf(Hf, lh);
    int ci = 0;
    for (auto j : Range(nffi)) {
      int base = BS * ffiinds[j];
      for (auto l : Range(BS))
        { colsi[ci++] = base++; }
    }
    for (auto kv : Range(HB))
      { colsi[ci++] = HA + kv; }
    int cf = 0;
    for (auto j : Range(nfff)) {
      int base = BS * fffinds[j];
      for (auto l : Range(BS))
        { colsf[cf++] = base++; }
    }

    // if (doco) cout << "Hi + Hf " << Hi << " " << Hf << " " << Hi+Hf << " " << HA << endl;
    // if (doco) cout << "HA HB HM " << HA << " " << HB << " " << HM << endl;
    // if (doco) cout << "Hc " << Hc << endl;

    // if (doco) {cout << "colsi (" << colsi.Size() << ") = "; prow(colsi); cout << endl; }
    // if (doco) {cout << "colsf (" << colsf.Size() << ") = "; prow(colsf); cout << endl;}

    FlatMatrix<double> Pf(Hf, Hc, lh); Pf = 0;
    for (auto j : Range(nfff)) {
      auto fnr = ffacets[fffinds[j]];
      for (auto l : Range(cfacets))
        { Pf.Rows(j*BS, (j+1)*BS).Cols(l*BS, (l+1)*BS) = const_P(f_e2dof[fnr], c_e2dof[cfacets[l]]); }
    }

    // if (doco) cout << "Pf: " << endl << Pf << endl;

    // if (doco) cout << "Bc: " << endl << Bc << endl;

    // if (doco) cout << "Bf: " << endl << Bf << endl;

    /** -A_if * P_f **/
    FlatMatrix<double> rhs(Hi + HB, Hc, lh);
    rhs = -M.Rows(colsi).Cols(colsf) * Pf;

    // if (doco) cout << "Mif " << endl << M.Rows(colsi).Cols(colsf) << endl;
    // if (doco) cout << "rhs only homogen. " << endl << rhs << endl;
    rhs.Rows(Hi, rhs.Height()) += Bc;

    // if (doco) cout << "full RHS: " << endl << rhs << endl;

    /** Lock constant pressure **/
    // auto Bf = M.Rows(HA, HA+HB).Cols(0, HA);
    // auto BfT = M.Rows(0, HA).Cols(HA, HA+HB);
    if ( lock_const_pressure ) {
      // for (auto k : Range(1)) {
      // for (auto k : Range(HB - 1)) {
        // M(HA + HB - 1, HA + k) = avg_da;
        // M(HA + k, HA + HB - 1) = avg_da;
      // }
      // entries should be \int_T q_i * p_bar = |T_i| * 1 * 1
      for (auto k : Range(HB - 1)) {
        M(HA + HB - 1, HA + k) = avg_da / avg_flow  * cvinv0 * fvd[agg_vs[k]].vol;
        M(HA + k, HA + HB - 1) = avg_da / avg_flow  * cvinv0 * fvd[agg_vs[k]].vol;
      }
      // pose avg(pressure) = 1
      rhs.Row(Hi + HB - 1) = avg_da/avg_flow * cvinv0 * cvol;
    }
    // Bf.Row(HB-1) = bfsum/((HB-1)*(HB-1));
    // BfT.Col(HB-1) = bfsum/((HB-1)*(HB-1));

    // if (doco) cout << "M mat: " << endl << M << endl;


    // TODO: DO I NEED THIS????
    // if (doco)
    // {
    //   FlatMatrix<double> Aii(Hi, Hi, lh), S(HB, HB, lh);
    //   auto iii = colsi.Part(0, Hi);
    //   auto bbb = colsi.Part(Hi);
    //   Aii = M.Rows(iii).Cols(iii);
    //   cout << "Aii " << endl << Aii << endl;
    //   CalcInverse(Aii);
    //   cout << "Aii inv " << endl << Aii << endl;
    //   S = M.Rows(bbb).Cols(bbb);
    //   S -= M.Rows(bbb).Cols(iii) * Aii * M.Rows(iii).Cols(bbb);
    //   cout << "S: " << endl << S << endl;
    //   CalcInverse(S);
    //   cout << "Sinv: " << endl << S << endl;
    // }

    /** The block to invert **/
    FlatMatrix<double> Mii(Hi + HB, Hi + HB, lh);
    Mii = M.Rows(colsi).Cols(colsi);

    // if (doco) cout << "Mii " << endl << Mii << endl;
    // if (doco) {
    //   cout << endl << "====" << endl;
    //   cout << " CALC RANK " << endl;
    //   cout << "====" << endl;
    //   print_rank2("Mii", Mii, lh);
    //   cout << endl << "====" << endl << endl;
    // }

    // cout << " BfT " << endl << Mii.Rows(0, Hi).Cols(Hi, Hi+HB) << endl;


    // REMOVE if no output!!
    // FlatMatrix<double> Mii_save (Mii.Height(), Mii.Height(), lh); Mii_save = Mii;

    CalcInverse(Mii);

    // {
    //   HeapReset hr(lh);
    //   const int N = Mii.Height();
    //   FlatMatrix<double> evecs(N, N, lh), evecs2(N, N, lh);
    //   FlatVector<double> evals(N, lh);
    //   LapackEigenValuesSymmetric(Mii, evals, evecs);
    //   evecs2 = evecs;
    //   for (auto & v : evals)
    //     if (abs(v) > 1e-12)
    //       { v = 1/v; }
    //     else
    //       { v = 0; }
    //   for (auto i : Range(N))
    //     for (auto j : Range(N))
    //       evecs(i,j) *= evals(i);
    //   Mii = Trans(evecs) * evecs2;
    // }

    // if (doco) cout << "Mii-inv " << endl << Mii << endl;

    // if (doco) {
    //   HeapReset hr(lh);
    //   const int N = Mii.Height();
    //   FlatMatrix<double> Id_please(N, N, lh);
    //   Id_please = Mii * Mii_save;
    //   cout << " Mii * Mii_save : " << endl << Id_please << endl;
    //   for (auto k : Range(N))
    //     { Id_please(k, k) -= 1.0; }
    //   for (auto k : Range(N))
    //     for (auto j : Range(N))
    //       if (abs(Id_please(k,j)) > 1e-10)
    // 	{ cout << " DIFF@" << k << " " << j << " = " << Id_please(k,j) << endl; }
    //   cout << endl;
    //   double err = L2Norm(Id_please);
    //   cout << " ERR 1 " << err << endl;
    //   Id_please = Mii_save * Mii;
    //   cout << " Mii_save * Mii : " << endl << Id_please << endl;
    //   for (auto k : Range(N))
    //     { Id_please(k, k) -= 1.0; }
    //   double err2 = L2Norm(Id_please);
    //   cout << " ERR 2 " << err2 << endl;
    //   for (auto k : Range(N))
    //     for (auto j : Range(N))
    //       if (abs(Id_please(k,j)) > 1e-10)
    // 	{ cout << " DIFF@" << k << " " << j << " = " << Id_please(k,j) << endl; }
    //   cout << endl;
    // }

    /** The final prol **/
    FlatMatrix<double> Pext(Hi + HB, Hc, lh);
    Pext = Mii * rhs;

    // if (doco) cout << "full Pext: " << endl << Pext << endl;

    // if (doco) {
    //   if (Pext.Width() > 5*BS) {
    //     cout << "Pext cols [5*BS, 6*BS): " << endl << Pext.Cols(5*BS, 6*BS) << endl;
    //   }
    // }

    /** Write into sprol **/
    for (auto kfi : Range(nffi)) {
        auto ff = ffacets[ffiinds[kfi]];
        auto fd = f_e2dof[ff];
        auto ris = P.GetRowIndices(fd);
        FlatVector<TM> rvs = P.GetRowValues(fd);
        // if (doco) cout << "write row " << kfi << " -> " << ff << " -> dof " << fd << endl;
        // if (doco) cout << "mat space = " << rvs.Size() << " * BS = " << rvs.Size() * BS << endl;
        // if (doco) cout << "Pext width = " << Pext.Width() << endl;
        for (auto j : Range(ncf)) // TODO??
          { ris[j] = c_e2dof[cfacets[j]]; }
        QuickSort(ris);
        for (auto j : Range(ncf)) {
          // rvs[j] = 0.0;
          rvs[ris.Pos(c_e2dof[cfacets[j]])] = Pext.Rows(BS*kfi, BS*(kfi+1)).Cols(BS*j, BS*(j+1));
        }
    }

    // if (doco) {
    //   // check div of all crs BFs
    //   cout << " ffacets "; prow2(ffacets); cout << endl;
    //   cout << " ffiinds "; prow2(ffiinds); cout << endl;
    //   cout << " fffinds "; prow2(fffinds); cout << endl;
    //   Array<double> divs(agg_vs.Size());
    //   for (auto j : Range(ncf)) {
    //     for (auto l : Range(BS)) {
    //       divs = 0.0;
    //       int cenr = cfacets[j], cdnr = c_e2dof[cenr];
    //       cout << endl << endl << " ====== " << endl << "check div of cdof " << j << "/" << ncf << " = e" << cenr << "/d" << cdnr << ", component " << l << endl;
    //       // cout << "c flow " << ced[cenr].flow << endl;
    //       auto Pcol = Pext.Col(BS*j+l);
    //       for (auto kv : Range(agg_vs)) {
    // 	auto vnr = agg_vs[kv];
    // 	// cout << " @vnr " << kv << " " << vnr << endl;
    // 	if (fvd[vnr].vol <= 0)
    // 	  { divs[kv] = -1.0; continue; }
    // 	divs[kv] = 0.0;
    // 	auto ecri = fecon.GetRowIndices(vnr);
    // 	auto ecrv = fecon.GetRowValues(vnr);
    // 	for (auto kj : Range(ecri)) {
    // 	  int vj = ecri[kj], posj = find_in_sorted_array(vj, agg_vs);
    // 	  int eij = int(ecrv[kj]), dij = f_e2dof[eij];
    // 	  int kij = find_in_sorted_array(eij, ffacets);
    // 	  int ki_fij = find_in_sorted_array(kij, ffiinds);
    // 	  int kf_fij = find_in_sorted_array(kij, fffinds);
    // 	  // cout << "  " << vj << "  " << posj << " " << eij << " " << dij << " " << kij << " " << ki_fij << " " << kf_fij << endl;
    // 	  double addval = 0.0;
    // 	  // cout << "  f flow " << fed[eij].flow << endl;
    // 	  if (ki_fij != -1) { // "free" interior edge
    // 	    for (auto kl : Range(BS))
    // 	      { addval += 1.0/fvd[vnr].vol * ( (fedges[eij].v[0]==vnr) ? 1.0 : -1.0) * (fed[eij].flow(kl) * Pcol(ki_fij*BS+kl)); }
    // 	  } else if (kf_fij != -1) { // "bnd" edge
    // 	    // cout << "  Pf block " << endl;
    // 	    // for (auto kl : Range(BS))
    // 	      // cout << "  Pf(" << BS*kf_fij+kl << " " << BS*j+l << ") = " << Pf(BS*kf_fij+kl, BS*j+l) << endl;
    // 	    for (auto kl : Range(BS))
    // 	      { addval += 1.0/fvd[vnr].vol * ( (fedges[eij].v[0]==vnr) ? 1.0 : -1.0) * (fed[eij].flow(kl) * Pf(BS*kf_fij+kl, BS*j+l)); }
    // 	  }
    // 	  if (addval) {
    // 	    // cout << "  -> add val " << addval << " to " << kv << ", old = " << divs[kv] << endl;
    // 	    divs[kv] += addval;
    // 	    // cout << "  is now " << divs[kv] << endl;
    // 	  }
    // 	} // ecri
    //       } // agg_vs
    //       cout << " CHECKED DIVS "; prow2(divs); cout << endl;
    //       for (auto j : Range(divs))
    // 	{ divs[j] *= fvd[agg_vs[j]].vol; }
    //       cout << " CHECKED int(DIVS) "; prow2(divs); cout << endl;
    //       double tot_int_div = 0.0, totvol = 0.0;
    //       for (auto j : Range(divs))
    // 	{ tot_int_div += divs[j]; totvol += fvd[agg_vs[j]].vol; }
    //       cout << " CHECKED td_int " << tot_int_div << " td_avgval " << tot_int_div/totvol << " totvol " << totvol << endl;
    //       cout << " div-avg_div = " << endl;
    //       for (auto j : Range(divs))
    // 	{ cout << divs[j]/fvd[agg_vs[j]].vol - tot_int_div/totvol << " "; }
    //       cout << " int(div-avg_div) = " << endl;
    //       for (auto j : Range(divs))
    // 	{ cout << divs[j] - fvd[agg_vs[j]].vol * tot_int_div/totvol << " "; }
    //     } // BS
    //   } // cdof
    // } // doco
  } // agglomerate loop

  // if (bdoco) {
  //   cout << "final stokes PWP: " << endl;
  //   print_tm_spmat(cout << endl, P);
  // }

  // if (O.log_level >= Options::LOG_LEVEL::DBG) {
  //   ofstream out ("stokes_pwp_rk" + to_string(FM.GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_" + to_string(fcap.baselevel) + ".out");
  //   print_tm_spmat(out, P);
  // }

  return prol;
} // NCStokesAMGFactory::BuildPrimarySpaceProlongation


template<class TMESH, class ENERGY>
void NCStokesAMGFactory<TMESH, ENERGY> :: BuildDivMat (StokesLevelCapsule& cap) const
{
  static Timer t("NCStokesAMGFactory::BuildDivMat"); RegionTimer rt(t);

  const auto & M = *static_pointer_cast<TMESH>(cap.mesh); M.CumulateData();
  const auto & eqc_h = *M.GetEQCHierarchy();

  /**
   *  Use range DOFs for ALL vertices, also ghost ones.
   *  Reason is that dofed edges can also have one ghost vertex.
   *  TODO: Rows without entries (can this happen?)
   *  TODO: would it not be better to have only solid verts??
   */
  shared_ptr<ParallelDofs> l2_pds = nullptr;

  if (cap.uDofs.IsParallel())
  {
    TableCreator<int> c_l2_dps(M.template GetNN<NT_VERTEX>());
    for (; !c_l2_dps.Done(); c_l2_dps++) {
      M.template ApplyEQ<NT_VERTEX>([&](auto eqc, auto v) {
        c_l2_dps.Add(v, eqc_h.GetDistantProcs(eqc));
      }, false);
    }
    auto l2_dps = c_l2_dps.MoveTable();
    l2_pds = make_shared<ParallelDofs>(eqc_h.GetCommunicator(), std::move(l2_dps), 1, false);
  }

  shared_ptr<TDM> div_mat;
  {
    auto vdata = get<0>(M.Data())->Data();
    auto edata = get<1>(M.Data())->Data();
    auto edges = M.template GetNodes<NT_EDGE>();
    auto [dofed_edges, dof2e, e2dof] = M.GetDOFedEdges();
    auto NE = M.template GetNN<NT_EDGE>(), NE_dofed = dof2e.Size();
    auto r_pds = cap.uDofs.GetParallelDofs();
    Array<int> perow(M.template GetNN<NT_VERTEX>()); perow = 0.0;
    for (auto k : Range(dof2e)) {
      auto enr = dof2e[k];
      perow[edges[enr].v[0]]++;
      perow[edges[enr].v[1]]++;
    }
    // cout << endl << " LEVEL " << cap.baselevel << " perow " << endl; prow2(perow); cout << endl << endl;
    div_mat = make_shared<TDM>(perow, NE_dofed);
    perow = 0;
    for (auto k : Range(dof2e)) {
      auto enr = dof2e[k];
      div_mat->GetRowIndices(edges[enr].v[0])[perow[edges[enr].v[0]]++] = k;
      div_mat->GetRowIndices(edges[enr].v[1])[perow[edges[enr].v[1]]++] = k;
    }
    for (auto k : Range(perow)) {
      auto ris = div_mat->GetRowIndices(k);
      // QuickSort(ris); // already sorted because previous loop goes through edge-dofs in ascending order
      auto rvs = div_mat->GetRowValues(k);
      // divide out cell volume. if volume is neg (fict. vertex) or zero (immersed vertex) mult with 0
      const double fac = (vdata[k].vol > 0.0) ? 1.0/vdata[k].vol : 0.0;
      // cout << " level " << cap.baselevel << " fill row " << k << ", vol = " << vdata[k].vol
            // << " with ris "; prow(ris); cout << endl;
      for (auto j : Range(ris)) {
        auto dnr = ris[j], enr = dof2e[dnr];
        double orient = (edges[enr].v[0] == k) ? 1.0 : -1.0;
        if ( (r_pds != nullptr) && (!r_pds->IsMasterDof(dnr)) )
          { orient = 0.0; } // this is a C2D operation!!
        for (auto l : Range(BS))
          { rvs[j](l) = orient * fac * edata[enr].flow(l); }
      }
    }
    // cout << " DIV MAT LEVEL " << cap.baselevel << ": " << endl;
    // print_tm_spmat(cout, *div_mat); cout << endl << endl;
  }

  cap.rr_uDofs = UniversalDofs(l2_pds, M.template GetNN<NT_VERTEX>(), 1);
  cap.div_mat = div_mat;
} // NCStokesAMGFactory::BuildDivMat


template<class TMESH, class ENERGY>
void NCStokesAMGFactory<TMESH, ENERGY> :: BuildCurlMat (StokesLevelCapsule& cap) const
{
  static Timer t("NCStokesAMGFactory::BuildCurlMat");
  RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);

  const auto & M = *static_pointer_cast<TMESH>(cap.mesh);
  M.CumulateData();

  auto loops = M.GetLoops();
  auto active_loops = M.GetActiveLoops();
  auto edata = get<1>(M.Data())->Data();
  auto [dofed_edges, dof2e, e2dof] = M.GetDOFedEdges();
  size_t ND = dof2e.Size();

  auto edges = M.template GetNodes<NT_EDGE>();

  // cout << " BuildCurlMat " << endl;
  // cout << " EDGES " << endl;
  // for (auto k : Range(edata)) {
  //   cout << edges[k] << " /// " << edata[k] << endl;
  // }
  // cout << endl;
  // cout << " dofed_edges " << endl << dofed_edges << endl;
  // cout << " dof -> edge " << endl; prow2(dof2e); cout << endl;
  // cout << " edge -> dof " << endl; prow2(e2dof); cout << endl;
  // cout << " datas " << dof2e.Data() << " " << e2dof.Data() << endl;

  Array<int> perow(loops.Size()); perow = 0;
  for (auto k : Range(loops.Size()))
    if ( (!active_loops) || (active_loops->Test(k)) )
for (auto ore : loops[k])
  if (dofed_edges->Test(abs(ore) - 1))
    { perow[k]++; }

  // cout << " loops " << endl << loops << endl;
  // cout << endl << endl;
  // cout << " perow " << endl << perow << endl;
  // cout << endl << endl;

  auto curlT_mat = make_shared<TCTM_TM>(perow, ND);

  for (auto k : Range(loops.Size())) {
    if ( active_loops && (!active_loops->Test(k)) )
      { continue; }
    auto loop = loops[k];
    auto ris = curlT_mat->GetRowIndices(k);
    auto rvs = curlT_mat->GetRowValues(k);
    int c = 0;
    for (auto j : Range(loop)) {
      int enr = abs(loop[j]) - 1;
      if (dofed_edges->Test(enr))
        { ris[c++] = e2dof[enr]; }
    }
    QuickSort(ris);
    for (auto j : Range(loop)) {
      int enr = abs(loop[j]) - 1;
      if (dofed_edges->Test(enr)) { // YES, loops include non-dofed, (gg-) edges
        int dnr = e2dof[enr];
        int col = ris.Pos(dnr);
        int fac = (loop[j] < 0) ? -1 : 1;
        auto flow = edata[enr].flow;
        // cout << " loop " << k << ", etr " << j << " " << loop[j] << " -> enr " << enr << ", dnr " << dnr << ", flow " << flow << endl;
        double fsum = 0;
        for (auto l : Range(BS))
          { fsum += sqr(flow[l]); }
        Mat<1, BS, double> &rval = rvs[col];
        for (auto l : Range(BS))
          { rval(0, l) = fac * flow[l]/fsum; }
      }
    }
  }

  auto curl_mat = TransposeSPM(*curlT_mat);

  // cout << " curl_mat " << curl_mat->Height() << " x " << curl_mat->Width() << endl;
  // print_tm_spmat(cout, *curl_mat);
  // cout << endl << endl;

  // cout << " curlT_mat " << curlT_mat->Height() << " x " << curlT_mat->Width() << endl;
  // print_tm_spmat(cout, *curlT_mat);
  // cout << endl << endl;

  cap.curl_mat_T = make_shared<TCTM>(std::move(*curlT_mat));
  cap.curl_mat = make_shared<TCM>(std::move(*curl_mat));

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    ofstream out ("stokes_CT_rk_" + to_string(M.GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_" + to_string(cap.baselevel) + ".out");
    // out << "MESH " << endl;
    // out << M << endl;
    // out << endl << endl;
    out << " EDATA " << endl;
    out << edata << endl << endl;
    out << " ECON " << endl;
    if (active_loops)  {
out << " ACTIVE_LOOPS " << endl;
for (auto j : Range(active_loops->Size()))
  { out << j << ": " << ( active_loops->Test(j) ? 1 : 0 ) << endl; }
out << endl;
    }
    out << *(M.GetEdgeCM()) << endl << endl;
    out << "CT-MAT " << endl;
    print_spmat(out, *cap.curl_mat_T);
  }

} // NCStokesAMGFactory::BuildCurlMat


/** Contract **/

template<class TMESH, class ENERGY>
shared_ptr<BaseDOFMapStep>
NCStokesAMGFactory<TMESH, ENERGY> ::
BuildContractDOFMap (shared_ptr<BaseGridMapStep> cmap,
                     shared_ptr<BaseAMGFactory::LevelCapsule> &pFCap,
                     shared_ptr<BaseAMGFactory::LevelCapsule> & b_mapped_cap) const
{
  static Timer t("BuildDOFContractMap"); RegionTimer rt(t);

  if (cmap == nullptr)
    { throw Exception("BuildDOFContractMap needs a mesh!"); }

  auto cm = my_dynamic_pointer_cast<StokesContractMap<TMESH>>(cmap,
              "NCStokesAMGFactory::BuildContractDOFMap - map");

  auto &fCap = *my_dynamic_pointer_cast<BaseStokesLevelCapsule>(pFCap,
                 "NCStokesAMGFactory::BuildContractDOFMap - fCap");

  auto mapped_cap = my_dynamic_pointer_cast<StokesLevelCapsule>(b_mapped_cap,
                    "NCStokesAMGFactory::BuildContractDOFMap - cap");

  fCap.savedCtrMap = cm;

  auto fmesh = my_dynamic_pointer_cast<TMESH>(cm->GetMesh(),
                "NCStokesAMGFactory::BuildContractDOFMap FMESH");

  auto &O = static_cast<Options&>(*options);

  if (O.log_level >= Options::LOG_LEVEL::DBG) {
    // Note: I think mapped_cap->baselevel is not (correctly) set here yet so we always write to the level-0 files here...
    ofstream outfm ("stokes_mesh_prectr_rk_" + to_string(cmap->GetMesh()->GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_" + to_string(mapped_cap->baselevel) + ".out");
    outfm << *fmesh << endl;

    ofstream out ("stokes_ctrmap_rk_" + to_string(cmap->GetMesh()->GetEQCHierarchy()->GetCommunicator().Rank()) + "_l_" + to_string(mapped_cap->baselevel) + ".out");
    out << *cm << endl;
  }

  // range DOF map
  auto fg = cm->GetGroup();
  Array<int> group(fg.Size()); group = fg;
  auto de_maps = CopyTable(cm->GetDofedEdgeMaps());

  mapped_cap->uDofs = (cm->IsMaster()) ? this->BuildUDofs(*mapped_cap) : UniversalDofs();

  auto ctr_map_range = make_shared<CtrMap<typename strip_vec<Vec<BS, double>>::type>> (fCap.uDofs,
                                                                                       mapped_cap->uDofs,
                                                                                       std::move(group),
                                                                                       std::move(de_maps));
  if (cm->IsMaster())
    { ctr_map_range->_comm_keepalive_hack = cm->GetMappedEQCHierarchy()->GetCommunicator(); }

  auto pot_fpd = fmesh->GetLoopUDofs().GetParallelDofs();
  group.SetSize(fg.Size()); group = fg;
  auto loop_maps = CopyTable(cm->GetLoopMaps());
  shared_ptr<ParallelDofs> pot_cpd = nullptr;

  if (cm->IsMaster()) {
    this->BuildPotUDofs(*mapped_cap);
    pot_cpd = mapped_cap->pot_uDofs.GetParallelDofs();
  }
  else
  {
    // dummy UniversalDofs
    mapped_cap->pot_uDofs = UniversalDofs();
  }

  if (false)
  {
    /**
     * I don't think we ever need to redistribute potential space vectors,
     * and that the Multi-step here was only used for the potential-space
     * AMG, which we don't/can't do anymore because we don't have commuting
     * potential space prolongations.
    */
    // pot DOF map

    auto ctr_map_pot = make_shared<CtrMap<typename strip_vec<double>::type>> (pot_fpd, pot_cpd, std::move(group), std::move(loop_maps));
    if (cm->IsMaster())
      { ctr_map_pot->_comm_keepalive_hack = cm->GetMappedEQCHierarchy()->GetCommunicator(); }

    Array<shared_ptr<BaseDOFMapStep>> step_comps(2);
    step_comps[0] = ctr_map_range;
    step_comps[1] = ctr_map_pot;
    auto multi_step = make_shared<MultiDofMapStep>(step_comps);

    return multi_step;
  }
  else
  {
    return ctr_map_range;
  }
} // NCStokesAMGFactory::BuildDOFContractMap


template<class TMESH, class ENERGY>
void
NCStokesAMGFactory<TMESH, ENERGY> ::
DoDebuggingTests (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
{
  auto &O = static_cast<Options&>(*options);
  // if (O.check_loop_divs)
    // { CheckLoopDivs(amg_levels, map); }
} // NCStokesAMGFactory::CheckKVecs


// TODO: inlined for linking, std::move implementation to a cpp file, definition to utils.hpp or sth
INLINE void SetUnitVec (shared_ptr<BaseVector> vec, int rank, int dof, double scale = 1.0, int bs = 1, int comp = 0)
{
  auto fv = vec->FVDouble(); fv = 0.0;
  if (auto parvec = dynamic_pointer_cast<ParallelBaseVector>(vec)) {
    auto comm = parvec->GetParallelDofs()->GetCommunicator();
    if (rank != 0) {
if (rank == comm.Rank())
  { fv(bs*dof+comp) = scale; }
vec->SetParallelStatus(DISTRIBUTED);
vec->Cumulate();
    }
    else if (comm.Size() == 2) {
if (rank == 1)
  { fv(bs*dof+comp) = scale; }
    }
    else {
vec->SetParallelStatus(CUMULATED);
auto pds = parvec->GetParallelDofs();
auto all = make_shared<BitArray>(pds->GetNDofLocal()); all->Set();
Array<int> gdn; int gn;
pds->EnumerateGlobally(all, gdn, gn);
for (auto k : Range(gdn))
  if (gdn[k] == dof)
    { fv(bs*k+comp) = 1.0; }
    }
  }
  else { fv(bs*dof+comp) = scale; }
}

template<class TMESH, class ENERGY>
void
NCStokesAMGFactory<TMESH, ENERGY> ::
CheckLoopDivs (FlatArray<shared_ptr<BaseAMGFactory::AMGLevel>> amg_levels, shared_ptr<DOFMap> map)
{
  auto gcomm = amg_levels[0]->cap->mesh->GetEQCHierarchy()->GetCommunicator();
  auto prtv = [&](auto & vec, string title, int bs) {
    auto fv = vec.FVDouble();
    cout << title << " = " << endl;
    cout << "  size = " << vec.Size() << " " << fv.Size() << endl;
    cout << "  bs = " << bs << endl;
    cout << "  stat = " << vec.GetParallelStatus() << endl;
    cout << "  vals = " << endl;
    for (auto k : Range(vec.Size())) {
bool nz = false;
for (auto l : Range(bs))
  if (abs(fv(k*bs+l)) > 1e-12)
    { nz = true; break; }
if (nz) {
  cout << k << ": ";
  for (auto l : Range(bs))
    { cout << fv(k*bs+l) << " "; }
  cout << endl;
}
    }
    cout << endl;
  };

  auto check_div = [&](int level, string name, shared_ptr<BaseVector> lvec) {
    cout << " check div of vec " << name << " from level " << level << endl;
    cout << " levels " << amg_levels << endl;
    gcomm.Barrier();
    int level_loc = int(amg_levels.Size())-1;
    if (amg_levels.Last()->cap->mat == nullptr) // contracted!
{ level_loc--; }
    level_loc = min(level, level_loc);
    cout << " levels : " << level << " " << amg_levels.Size()-1 << " " << level_loc << endl;
    cout << " last mat " << amg_levels.Last()->cap->mat << endl;
    Array<shared_ptr<BaseVector>> r_vecs(level+1);
    for (auto k : Range(min(level, int(amg_levels.Size()))))
{ cout << " get vec " << k << endl; r_vecs[k] = map->CreateVector(k); }
    cout << " got vecs " << endl;
    if (level < amg_levels.Size())
{ r_vecs.Last() = lvec; }
    cout << " got crst vec " << endl;
    cout << " r_vecs " << endl << r_vecs << endl;
    gcomm.Barrier();
    /** get [0..level) vecs **/
    // need to count "empty" level from contr for vec transfers
    int level_loc_tr = min(int(amg_levels.Size())-1, level);
    for (int k = level_loc_tr-1; k >= 0; k--) // A->B, vec_A, vec_B
{ map->TransferAtoB(k+1, k, r_vecs[k+1].get(), r_vecs[k].get()); }
    cout << " transed vecs ( " << level_loc << " " << level_loc << endl;
    // prtv(*r_vecs.Last(), "AT r_vec", BS);
    /** check divs **/
    gcomm.Barrier();
    for (int k = level_loc; k >= 0; k--) {
      cout << endl << " check div of vec " << name << " from level " << level << " on level " << k << ": " << endl;
      auto slc_cap = static_pointer_cast<StokesLevelCapsule>(amg_levels[k]->cap);
      const auto & cap = static_cast<const StokesLevelCapsule&>(*amg_levels[k]->cap);
      auto pds = map->GetParDofs(k);
      cout << " baselev " << cap.baselevel << endl;
      cout << " mat " << cap.mat << endl;
      cout << " pds "  << pds << endl;
      cout << " pds2 " << cap.uDofs.GetParallelDofs() << endl;
      cout << " pds3 " << cap.pot_uDofs.GetParallelDofs() << endl;
      cout << " pds4 " << cap.rr_uDofs.GetParallelDofs() << endl;
      cout << " mat name " << typeid(cap.mat).name() << endl;
      pds->GetCommunicator().Barrier();
      auto div_mat = cap.div_mat;
      cout << " div_mat " << div_mat << endl;
      shared_ptr<BaseMatrix> p_div_mat = div_mat;
      if (cap.rr_uDofs.IsParallel()) // TODO: should this not be rr_uDofs and uDofs instea dof rr_uDofs twice??
       p_div_mat = make_shared<ParallelMatrix>(p_div_mat, cap.rr_uDofs.GetParallelDofs(), cap.rr_uDofs.GetParallelDofs(), PARALLEL_OP::C2D);
      r_vecs[k]->Cumulate();
      auto rr_vec = p_div_mat->CreateColVector();
      (*rr_vec) = (*p_div_mat) * (*r_vecs[k]);
      (*rr_vec).Cumulate(); // autovector
      auto fv_r = r_vecs[k]->FV<Vec<BS, double>>();
      auto fv = (*rr_vec).FVDouble();
      double eps = 1e-8;
      const auto & M = static_cast<const TMESH&>(*cap.mesh);
      auto [dofed_edges, dof2e, e2dof] = M.GetDOFedEdges();
      auto edges = M.template GetNodes<NT_EDGE>();
      auto vdata = get<0>(M.Data())->Data();
      auto edata = get<1>(M.Data())->Data();
      cout << " ran vec sz " << r_vecs.Size() << endl;
      cout << " rvk " << r_vecs[k] << endl;
      // prtv(*r_vecs[k], "ran vec "+to_string(k), BS);
      for (auto vnr : Range(fv)) {
        if (abs(fv(vnr)) > eps) {
          cout << endl << " div on level " << k << ", vert " << vnr << " = " << fv(vnr) << ", vol = " << vdata[vnr].vol << endl;
          cout << " vert shared with = "; prow(cap.rr_uDofs.GetDistantProcs(vnr)); cout << endl;
          cout << " dnr/enr/r_val/div_val/div_val*r_val/edge_flow/edge_flow*r_val/dof-dps: " << endl;
          auto div_cols = div_mat->GetRowIndices(vnr);
          // auto div_vals = div_mat->GetRowValues(vnr);
          for (auto j : Range(div_cols)) {
            auto dnr = div_cols[j], enr = dof2e[dnr];
            throw Exception("Need to Dispatch over div-mat width here!");
            // double div_vf = (div_vals[j] * fv_r(dnr))(0); // result is Vec<1, double>
            // if (abs(div_vf) < 1e-8)
              // { cout << " SKIP edge " << edges[enr] << endl; continue; }
            double flow_vf = 0.0;
            for (auto l : Range(BS))
              { flow_vf += edata[enr].flow(l) * fv_r(dnr)(l); }
            if ( edges[enr].v[0] != vnr)
              { flow_vf = -flow_vf; }
            cout << " edge " << edges[enr] << ",";
            cout << " pds:: "; prow(pds->GetDistantProcs(dnr)); cout << endl;
            cout << j << " of " << div_cols.Size() << ": " << dnr << " / " << enr << " / " << fv_r(dnr) << endl;
            // cout << " divval::" << div_vals[j] << " / div::" << div_vf << endl;
            cout << " eflow:: " << edata[enr].flow << " / flow::" << flow_vf << " / ";
            cout << endl;
          }
        }
      }
    }
  };

  auto check_loop_div = [&](int rank, int level, int lnr) {
    /** get "level" range vector **/
    shared_ptr<BaseVector> r_vec;
    if (level < amg_levels.Size()) {
      auto c_mat       = static_cast<const StokesLevelCapsule&>(*amg_levels[level]->cap).curl_mat;
      auto pot_uDofs   = static_cast<const StokesLevelCapsule&>(*amg_levels[level]->cap).pot_uDofs;
      auto range_uDofs = static_cast<const StokesLevelCapsule&>(*amg_levels[level]->cap).uDofs;
      cout << " mat p r pds " << c_mat << " " << pot_uDofs << " " << range_uDofs << endl;
      // shared_ptr<BaseMatrix> par_c_mat = c_mat;
      // if (pot_uDofs.IsParallel())
      //  { par_c_mat = make_shared<ParallelMatrix>(c_mat, pot_uDofs.GetParallelDofs(), range_uDofs.GetParallelDofs(), PARALLEL_OP::C2C); }
      auto par_c_mat = WrapParallelMatrix(c_mat, pot_uDofs, range_uDofs, PARALLEL_OP::C2C);
      /** Get pot. space vec on correct level with one entry at correct loop # **/
      auto pot_vec = par_c_mat->CreateRowVector();
      SetUnitVec(pot_vec, rank, lnr, 1.0, 1, 0);
      /** range vector **/
      r_vec = par_c_mat->CreateColVector();
      *r_vec = 0.0;
      (*pot_vec).Cumulate();
      c_mat->Mult(*pot_vec, *r_vec);
      r_vec->Cumulate();
      cout << " rvl " << r_vec << endl;
      // prtv(*pot_vec, "pot_vec", 1);
      // prtv(*r_vec, "r_vec", BS);
    }
    check_div(level, "loop rk "+to_string(rank)+", lev "+to_string(level)+", lnr " + to_string(lnr), r_vec);
  };

  {
    for (auto l : Range(BS)) {
shared_ptr<BaseVector> dof_vec;
if (amg_levels.Size() > 2) {
  dof_vec = map->CreateVector(2);
  SetUnitVec(dof_vec, 2, 601, 1.0, BS, l);
}
check_div(2, "dof P2L2E699C"+to_string(l), dof_vec);
    }
    // the one below makes sense i think
    for (auto l : Range(BS)) {
shared_ptr<BaseVector> dof_vec;
if (amg_levels.Size() > 2) {
  dof_vec = map->CreateVector(2);
  SetUnitVec(dof_vec, 2, 112, 1.0, BS, l);
}
check_div(2, "dof P2L2D150C"+to_string(l)+"(comp)", dof_vec);
    }
  }

  check_loop_div(0, 2, 1870);
  check_loop_div(0, 2, 1932);
  check_loop_div(0, 2, 1954);
  check_loop_div(0, 2, 1955);
  check_loop_div(0, 2, 2026);

  check_loop_div(0, 2, 665);

} // NCStokesAMGFactory::ChekLoopDivs


template<class TMESH, class ENERGY>
NCStokesAMGFactory<TMESH, ENERGY>::RTZToBlockEmbedding::
RTZToBlockEmbedding (UniversalDofs uDofs)
  : BASE(nullptr,
          uDofs,
          UniversalDofs(uDofs.GetParallelDofs(),
                        uDofs.GetND(),
                        1))
  , _presComp(uDofs.GetND())
{

} // RTZToBlockEmbedding(..)


template<class TMESH, class ENERGY>
void
NCStokesAMGFactory<TMESH, ENERGY>::RTZToBlockEmbedding::
Finalize ()
{
  /**
   * this is needed only on AMG level 0, we KNOW there are no zero-flow
   * edges there, so just discard the freedofs here
  */
  FinalizeWithFD();
} // RTZToBlockEmbedding::Finalize

template<class TMESH, class ENERGY>
shared_ptr<BitArray>
NCStokesAMGFactory<TMESH, ENERGY>::RTZToBlockEmbedding::
FinalizeWithFD ()
{
  shared_ptr<BitArray> secFree = nullptr;

  if (this->GetProl() == nullptr)
  {
    secFree = make_shared<BitArray>(this->GetUDofs().GetND());

    secFree->Clear();

    for (auto k : Range(_presComp))
    {
      auto &vK = GetPresComp(k);

      double const norm = L2Norm(vK);

      if (norm > 0)
      {
        vK /= norm;
        secFree->SetBit(k);
      }
    }

    SetUpProl();

    BASE::Finalize();
  }

  return secFree;
} // RTZToBlockEmbedding::FinalizeWithFD


template<class TMESH, class ENERGY>
void
NCStokesAMGFactory<TMESH, ENERGY>::RTZToBlockEmbedding::
SetPreservedComponent (size_t          const &k,
                       Vec<BS, double> const &vec)
{
  _presComp[k] = vec;
} // RTZToBlockEmbedding::SetPreserevedComponent


template<class TMESH, class ENERGY>
template<bool IS_RIGHT, int BSA, int BSB>
INLINE shared_ptr<SparseMatrix<double>>
NCStokesAMGFactory<TMESH, ENERGY>::RTZToBlockEmbedding::
ProjectMatrixImpl (SparseMatTM<BSA, BSB> const &A,
                   RTZToBlockEmbedding   const *rightEmb) const
{
  static_assert( ( (BSA == BS) && (BSB == BS) ) ||
                 ( (BSA == 1 ) && (BSB == BS) ) ||
                 ( (BSA == BS) && (BSB == 1 ) ) );

  if constexpr(!IS_RIGHT)
  {
    if (rightEmb == nullptr)
    {
      throw Exception("RTZToBlockEmbedding::ProjectMatrixImpl - NEED RIGHT EMB!");
    }
  }

  // this is simpler than true sparse matrix multiplication!
  MatrixGraph const &graphA = A;

  auto pB = make_shared<SparseMatrix<double>>(graphA);

  auto const &B = *pB;

  auto getVJ = [&](auto J)
  {
    if constexpr(IS_RIGHT)
    {
      return GetPresComp(J);
    }
    else
    {
      return rightEmb->GetPresComp(J);
    }
  };

  for (auto k : Range(A.Height()))
  {
    auto ris = A.GetRowIndices(k);
    auto rvA = A.GetRowValues(k);
    auto rvB = B.GetRowValues(k);

    for (auto j : Range(rvA))
    {
      if constexpr ( BSB == 1 ) // -> BSA == BS
      {
        /**
         * -> BSA == BS, A is (BSx1) curl-matrix, we are computing restriction * curl-mat,
         *      B = leftPT * A,   x = vT A
         */
        // Curl-mat restriction, B = PT * A -> x = vT A
        auto const &vecK = GetPresComp(k);
        // rvB[j] = (Trans(rvA[j]) * vecK)(0); // assign Vec<1, double> -> double
        rvB[j] = InnerProduct(vecK, rvA[j]);
      }
      else if constexpr ( BSA == 1)
      {
        /**
         * -> BSB == BS, A is (1xBS) curl-matrix trans, we are computing curl * prol,
         *      B = A * leftP,   x = A v
         */
        auto const &vecJ = getVJ(ris[j]);

        rvB[j] = InnerProduct(rvA[j], vecJ);
      }
      else
      {
        /**
         * -> BSA == BS == BSB, A is either (BSxBS) range-mat or (BSxBS) range-prol,
         *    we are either computing projected A-mat or Prol,
         *      B = leftPT * A * rightP
         */
        auto const &vecK = GetPresComp(k);
        auto const &vecJ = getVJ(ris[j]);

        rvB[j] = InnerProduct(vecK, rvA[j] * vecJ);
      }
    }
  }

  return pB;
} // RTZToBlockEmbedding::ProjectMatrixImpl


template<class TMESH, class ENERGY>
template<int BSA, int BSB>
shared_ptr<SparseMatrix<double>>
NCStokesAMGFactory<TMESH, ENERGY>::RTZToBlockEmbedding::
ProjectMatrix (SparseMatTM<BSA, BSB> const &A) const
{
  return ProjectMatrixImpl<true, BSA, BSB>(A);

  // // this is simpler than true sparse matrix multiplication!
  // shared_ptr<MatrixGraph> graphA = A;

  // auto B = make_shared<SparseMatrix<double>>(graphA);

  // auto const &ncA = *A;
  // auto const &ncB = *B;

  // for (auto k : Range(ncA.Height()))
  // {
  //   auto ris = ncA.GetRowIndices(k);
  //   auto rvA = ncA.GetRowValues(k);
  //   auto rvB = ncB.GetRowValues(k);

  //   auto const &vecK = GetPresComp(k);

  //   for (auto j : Range(rvA))
  //   {
  //     if constexpr ( BSB == 1 )
  //     {
  //       // Curl-mat restriction, B = PT * A -> x = vT A
  //       rvB[j] = Trans(rvA[j]) * vecK; // InnerProduct(vecK, rvA[j]);
  //     }
  //     else
  //     {
  //       auto const col = ris[j];
  //       auto const &vecJ = GetPresComp(col);

  //       if constexpr( (BSB == BS) && (BSC == BS) )
  //       {
  //         // Galerkin-Projection B = PT * A * P -> x = vT A v
  //         rvB[j] = InnerProduct(vecK, rvA[j] * vecJ);
  //       }
  //       else
  //       {
  //         // CurlT-mat prol, B = A * P -> x = A v
  //         rvB[j] = InnerProduct(vecK, rvA[j]);
  //       }
  //     }
  //   }
  // }

  // return B;
} // RTZToBlockEmbedding::ProjectMatrix


template<class TMESH, class ENERGY>
shared_ptr<SparseMatrix<double>>
NCStokesAMGFactory<TMESH, ENERGY>::RTZToBlockEmbedding::
ProjectProl (SparseMatTM<BS, BS> const &P,
             RTZToBlockEmbedding const &rightEmb) const
{
  return ProjectMatrixImpl<false,BS,BS>(P, &rightEmb);
  // this is simpler than true sparse matrix multiplication!
  // shared_ptr<MatrixGraph> graphP = P;

  // auto B = make_shared<SparseMatrix<double>>(graphP);

  // auto const &ncP = *P;
  // auto const &ncB = *B;

  // for (auto k : Range(ncP.Height()))
  // {
  //   auto ris = ncP.GetRowIndices(k);
  //   auto rvP = ncP.GetRowValues(k);
  //   auto rvB = ncB.GetRowValues(k);

  //   auto const &vecK = GetPresComp(k);

  //   for (auto j : Range(rvP))
  //   {

  //     auto const col   = ris[j];
  //     auto const &vecJ = rightEmb.GetProsComp(col);

  //     // Galerkin-Projection B = PT * A * P -> x = vT A v
  //     rvB[j] = InnerProduct(vecK, rvP[j] * vecJ);
  //   }
  // }

  // return B;
} // RTZToBlockEmbedding::ProjectProl


template<class TMESH, class ENERGY>
void
NCStokesAMGFactory<TMESH, ENERGY>::
RTZToBlockEmbedding::
SetUpProl ()
{
  auto const N = this->GetUDofs().GetND();

  Array<int> perow(N);
  perow = 1;

  this->_prol  = make_shared<SparseMat<BS, 1>>(perow, N);
  this->_prolT = make_shared<SparseMat<1, BS>>(perow, N);

  auto const &P  = *this->GetProl();
  auto const &PT = *this->GetProlTrans();

  for (auto k : Range(N))
  {
    auto const &vec = _presComp[k];

    P.GetRowIndices(k) = k;
    PT.GetRowIndices(k) = k;

    auto &v  = P.GetRowValues(k)[0];
    auto &vT = PT.GetRowValues(k)[0];

    Iterate<BS>([&](auto l)
    {
      v(l.value, 0)  = vec(l.value);
      vT(0, l.value) = vec(l.value);
    });
  }
} // RTZToBlockEmbedding::SetUpProl


template<class TMESH, class ENERGY>
std::tuple<shared_ptr<BaseProlMap>,
            shared_ptr<BaseStokesLevelCapsule>>
NCStokesAMGFactory<TMESH, ENERGY>::
CreateRTZLevel (BaseStokesLevelCapsule const &primCap) const
{
  /**
   *  We never need an RTZ level on AMG-level 0!
   */
  if (primCap.baselevel == 0)
  {
    throw Exception("Called into CreateRTZLevel on AMG-level 0!");
  }

  auto pCap = make_shared<StokesLevelCapsule>();
  auto &cap = *pCap;

  auto embStep = CreateRTZEmbeddingImpl(primCap);

  // mesh - do we need it??
  cap.baselevel = primCap.baselevel;

  // primCap.mesh = cap.mesh;

  if (primCap.free_nodes)
  {
    throw Exception("cap.free_nodes != nullptr in NC-FES CreateRTZLevel!");
  }

  cap.free_nodes = embStep->FinalizeWithFD();

  // embedding, fine:mesh<-coarse:RTZ, so use mapped UDofs
  cap.uDofs = embStep->GetMappedUDofs();

  // range-matrix
  auto const &primA = *my_dynamic_cast<SparseMat<BS, BS> const>(primCap.mat.get(),
                        "NCStokesAMGFactory::CreateRTZLevel - mat");

  cap.mat = embStep->template ProjectMatrix<BS, BS>(primA);

  // cout << " RTZ-matrix = " << cap.mat << ": " << cap.mat->Height() << " x " << cap.mat->Width() << endl;

  /**
   *  We have the same potential space, and the curl-matrix
   *   pot->sec is just pot->prim->sec, using the
   *  prim->sec restriction  (i.e. inverse of sec->prim prol)
   */
  cap.pot_mat      = primCap.pot_mat;
  cap.pot_freedofs = primCap.pot_freedofs;
  cap.pot_uDofs    = primCap.pot_uDofs;

  auto const &primC  = *my_dynamic_cast<SparseMat<BS, 1> const>(primCap.curl_mat.get(),
                          "NCStokesAMGFactory::CreateRTZLevel - C-mat");

  cap.curl_mat   = embStep->template ProjectMatrix<BS, 1>(primC);

  auto const &primCT = *my_dynamic_cast<stripped_spm_tm<Mat<1, BS>> const>(primCap.curl_mat_T.get(),
                          "NCStokesAMGFactory::CreateRTZLevel - CT-mat");

  cap.curl_mat_T = embStep->template ProjectMatrix<1, BS>(primCT);

  return std::make_tuple(embStep, pCap);
} // NCStokesAMGFactory::CreateRTZLevel


template<class TMESH, class ENERGY>
shared_ptr<typename NCStokesAMGFactory<TMESH, ENERGY>::RTZToBlockEmbedding>
NCStokesAMGFactory<TMESH, ENERGY>::
CreateRTZEmbeddingImpl (BaseStokesLevelCapsule const &cap) const
{
  auto const &TM = *my_dynamic_pointer_cast<TMESH>(cap.mesh, "CreateRTZEmbedding");

  auto [dOFedEdges, dofe2e_SB, e2dofe_SB] = TM.GetDOFedEdges();
  auto &dofe2e     = dofe2e_SB;
  auto &e2dofe     = e2dofe_SB;

  auto const nEdges = TM.template GetNN<NT_EDGE>();
  auto const nRTZ   = dofe2e.Size();

  auto prolStep = make_shared<RTZToBlockEmbedding>(cap.uDofs);

  auto &attEdgeData = *get<1>(TM.Data());
  auto  edgeData    = attEdgeData.Data();

  attEdgeData.Cumulate();

  TM.template Apply<NT_EDGE>([&](auto const &edge)
  {
    auto const dofNum = e2dofe[edge.id];

    if (dofNum != -1)
    {
      prolStep->SetPreservedComponent(dofNum, edgeData[edge.id].flow);
    }
  });

  return prolStep;
} // NCStokesAMGFactory::CreateRTZEmbeddingImpl


template<class TMESH, class ENERGY>
shared_ptr<BaseDOFMapStep>
NCStokesAMGFactory<TMESH, ENERGY>::
CreateRTZEmbedding (BaseStokesLevelCapsule const &cap) const
{
  auto embStep = CreateRTZEmbeddingImpl(cap);

  embStep->Finalize();

  return embStep;
}


template<class TMESH, class ENERGY>
shared_ptr<BaseDOFMapStep>
NCStokesAMGFactory<TMESH, ENERGY>::
CreateRTZDOFMap (BaseStokesLevelCapsule const &fCap,
                 BaseDOFMapStep         const &fEmb,
                 BaseStokesLevelCapsule const &cCap,
                 BaseDOFMapStep         const &cEmb,
                 BaseDOFMapStep         const &dOFStep) const
{
  BaseDOFMapStep const *meshDOFStep;

  if (fCap.baselevel == 0)
  {
    // on level 0, "dOFStep" is concatenation of embedding, the first prolongation
    // (and potentially first contract-map, but that does not work yet anyways!)
    auto const &savedMaps = fCap.savedDOFMaps;

    meshDOFStep = savedMaps.Last().get();
  }
  else
  {
    meshDOFStep = &dOFStep;
  }

  auto const &prolStep = *my_dynamic_cast<ProlMap<StripTM<BS, BS>> const>(meshDOFStep,
                    "NCStokesAMGFactory::CreateRTZDOFMap - DOFStep I");

  // std::cout << "fEmb: " << typeid(fEmb).name() << endl;

  auto const &fEmbTM = *my_dynamic_cast<RTZToBlockEmbedding const>(&fEmb,
                    "NCStokesAMGFactory::CreateRTZDOFMap - DOFStep II");

  // std::cout << "cEmb: " << typeid(cEmb).name() << endl;

  auto const &cEmbTM = *my_dynamic_cast<RTZToBlockEmbedding const>(&cEmb,
                    "NCStokesAMGFactory::CreateRTZDOFMap - DOFStep III");

  auto secProl = fEmbTM.ProjectProl(*prolStep.GetProl(),
                                    cEmbTM);

  auto secProlMap = make_shared<ProlMap<double>>(secProl,
                                                 fEmb.GetMappedUDofs(),  // "fine":   the mapped UD are sec-space
                                                 cEmb.GetMappedUDofs()); // "coarse": the mapped UD are sec-space

  secProlMap->BuildPT();

  secProlMap->Finalize();

  return secProlMap;
} // NCStokesAMGFactory::CreateRTZDOFMap

  /** END NCStokesAMGFactory **/

} // namespace amg

#endif // FILE_AMG_FACTORY_STOKES_NC_IMPL_HPP


