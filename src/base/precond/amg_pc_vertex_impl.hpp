#ifndef FILE_AMG_PC_VERTEX_IMPL_HPP
#define FILE_AMG_PC_VERTEX_IMPL_HPP

#include <fespace.hpp>
namespace amg
{
/** Options **/

extern bool CheckBAConsistency(string name, shared_ptr<BitArray> ba, shared_ptr<ParallelDofs> pds);

template<class FACTORY>
class VertexAMGPC<FACTORY> :: Options : public FACTORY::Options,
                                        public VertexAMGPCOptions
{
public:
  virtual void SetFromFlags (shared_ptr<FESpace> fes, shared_ptr<BaseMatrix> finest_mat, const Flags & flags, string prefix) override
  {
    FACTORY::Options::SetFromFlags(flags, prefix);
    VertexAMGPCOptions::SetFromFlags(fes, finest_mat, flags, prefix);
  }
}; // VertexAMGPC::Options

/** END Options **/


/** VertexAMGPC **/

template<class FACTORY>
VertexAMGPC<FACTORY> :: VertexAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
  : BaseAMGPC(blf, flags, name, opts)
{
  ;
} // VertexAMGPC(..)

template<class FACTORY>
VertexAMGPC<FACTORY> :: VertexAMGPC (shared_ptr<BaseMatrix> A, Flags const &flags, const string name, shared_ptr<Options> opts)
  : BaseAMGPC(A, flags, name, opts)
{
  ;
} // VertexAMGPC(..)

template<class FACTORY>
VertexAMGPC<FACTORY> :: ~VertexAMGPC ()
{
  ;
} // ~VertexAMGPC


template<class FACTORY>
void VertexAMGPC<FACTORY> :: InitLevel (shared_ptr<BitArray> freedofs)
{
  BaseAMGPC::InitLevel(freedofs);

  auto & O(static_cast<Options&>(*options));

  if (freedofs == nullptr) // postpone to FinalizeLevel
    { return; }

  if (O.spec_ss == VertexAMGPCOptions::SPECIAL_SUBSET::SPECSS_FREE)
  {
    if (O.log_level_pc > Options::LOG_LEVEL_PC::BASIC)
      { cout << IM(3) << "taking subset for coarsening from freedofs" << endl; }

    O.ss_select = finest_freedofs;
    //cout << " freedofs (for coarsening) set " << options->ss_select->NumSet() << " of " << options->ss_select->Size() << endl;
    //cout << *options->ss_select << endl;
    //CheckBAConsistency("ss_select", O.ss_select, bfa->GetFESpace()->GetParallelDofs());
  }

} // VertexAMGPC<FACTORY>::InitLevel


template<class FACTORY>
shared_ptr<BaseAMGPC::Options> VertexAMGPC<FACTORY> :: NewOpts ()
{
  return make_shared<Options>();
}


template<class FACTORY>
void VertexAMGPC<FACTORY> :: SetOptionsFromFlags (BaseAMGPC::Options& _O, const Flags & flags, string prefix)
{
  Options* myO = dynamic_cast<Options*>(&_O);
  if (myO == nullptr)
    { throw Exception("Invalid Opts!"); }
  Options & O(*myO);

  O.SetFromFlags(strict_alg_mode ? nullptr : bfa->GetFESpace(), finest_mat, flags, prefix);
} // VertexAMGPC<FACTORY> :: SetOptionsFromFlags


template<class FACTORY>
shared_ptr<EQCHierarchy> VertexAMGPC<FACTORY> :: BuildEQCH ()
{
  auto & O = static_cast<Options&>(*options);

  /** Build inital EQC-Hierarchy **/
  auto fpd = finest_mat->GetParallelDofs();

  // size_t maxset = 0;
  // this is garbage for SELECTED_SUBSET: say proc K has 0 selected DOFs, proc J has a bit set in a dof
  // that is larger than one shared with proc K -> leads to inconsistent initial EQCs
  // switch (O.subset) {
  // case(Options::DOF_SUBSET::RANGE_SUBSET): { maxset = O.ss_ranges.Last()[1]; break; }
  // case(Options::DOF_SUBSET::SELECTED_SUBSET): {
  //   auto sz = O.ss_select->Size();
  //   for (auto k : Range(sz))
  // 	if (O.ss_select->Test(--sz))
  // 	  { maxset = sz+1; break; }
  //   break;
  // } }
  // maxset = min2(maxset, fpd->GetNDofLocal());
  // shared_ptr<EQCHierarchy> eqc_h = make_shared<EQCHierarchy>(fpd, true, maxset);

  size_t maxset = 0;
  shared_ptr<EQCHierarchy> eqc_h;
  if (fpd == nullptr)
    { eqc_h = make_shared<EQCHierarchy>(); } // dummy EQCH
  else {
    size_t ndLoc = fpd->GetNDofLocal();
    switch (O.subset) {
      case(Options::DOF_SUBSET::RANGE_SUBSET): {
        maxset = min2(O.ss_ranges.Last()[1], ndLoc);
        eqc_h = make_shared<EQCHierarchy>(fpd, true, maxset);
        break;
      }
      case(Options::DOF_SUBSET::SELECTED_SUBSET): {
        auto sz = O.ss_select->Size();
        for (auto k : Range(sz))
        if (O.ss_select->Test(--sz))
          { maxset = sz+1; break; }
        maxset = min2(maxset, ndLoc);
        eqc_h = make_shared<EQCHierarchy>(fpd, true, maxset, O.ss_select);
        break;
      }
    }
  }

  return eqc_h;
} // VertexAMGPC::BuildEQCH


template<class FACTORY>
shared_ptr<TopologicMesh> VertexAMGPC<FACTORY> :: BuildInitialMesh ()
{
  static Timer t("BuildInitialMesh"); RegionTimer rt(t);

  auto eqc_h = BuildEQCH();

  // TODO: it is kinda dumb that we have the subset being decided in SetFromFlags,
  //       which in some cases requires looping through all kinds of stuff already,
  //       and then we have to loop through stuff in SetUpMaps again.
  //       I guess we should do both already when the PC is created

  /**
   *  This computes initial, local maps for the AMG-embedding
   *      - it creates the subset, i.e. the range of the embedding
   *      - it creates a mapping DOF -> prelim-vertex
   *  These are only preliminary, LOCAL maps
   */
  SetUpMaps();

  /**
   *  This creates an AlgMesh, in parallel this involves computing a
   *  re-ordering of the prelim-vertices implicitly defined by the above
   *  computed dof->prelim-vertex map.
   *
   *  That is, node_sort is the prelim-vertex -> vertex map
   */
  auto mesh = BuildAlgMesh(BuildTopMesh(eqc_h));

  return mesh;
} // VertexAMGPC::BuildInitialMesh


template<class FACTORY>
void VertexAMGPC<FACTORY> :: SetUpMaps ()
{
  static Timer t("SetUpMaps"); RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);

  const size_t ndof = strict_alg_mode ? finest_mat->Height() : bfa->GetFESpace()->GetNDof();

  size_t n_verts = -1;

  use_p2_emb = false;

  switch(O.subset) {
  case(Options::DOF_SUBSET::RANGE_SUBSET): {

    size_t in_ss = 0;
    for (auto range : O.ss_ranges)
      { in_ss += range[1] - range[0]; }

    switch(O.dof_ordering)
    {
      case(VertexAMGPCOptions::DOF_ORDERING::VARIABLE_ORDERING):
      {
        throw Exception("VARIABLE_ORDERING not implemented (how did we get here anyways?)");
        break;
      }
      case(VertexAMGPCOptions::DOF_ORDERING::REGULAR_ORDERING):
      {
        const size_t dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
        n_verts = in_ss / dpv;

        d2v_array.SetSize(ndof); d2v_array = -1;

        auto n_block_types = O.block_s.Size();

        if (dpv == 1) { // range subset , regular order, 1 dof per V
          v2d_array.SetSize(n_verts);
          int c = 0;
          for (auto range : O.ss_ranges) {
            for (auto dof : Range(range[0], range[1])) {
              d2v_array[dof] = c;
              v2d_array[c++] = dof;
            }
          }
        }
        else if (n_block_types == 1) { // range subset, regular order, N dofs per V in a single block
          use_v2d_tab = true;
          v2d_table = Table<int>(n_verts, dpv);
          auto v2da = v2d_table.AsArray();
          int c = 0;
          for (auto range : O.ss_ranges) {
            for (auto dof : Range(range[0], range[1])) {
              d2v_array[dof] = c / dpv;
              v2da[c++] = dof;
            }
          }
        }
        else { // range subset , regular order, N dofs per V in multiple blocks
          use_v2d_tab = true;
          v2d_table = Table<int>(n_verts, dpv);
          const int num_block_types = O.block_s.Size();
          int block_type = 0; // we currently mapping DOFs in O.block_s[block_type]-blocks
          int cnt_block = 0; // how many of those blocks have we gone through
          int block_s = O.block_s[block_type];
          int bos = 0;
          for (auto range_num : Range(O.ss_ranges)) {
            IVec<2,size_t> range = O.ss_ranges[range_num];
            while ( (range[1] > range[0]) && (block_type < num_block_types) ) {
              int blocks_in_range = (range[1] - range[0]) / block_s; // how many blocks can I fit in here ?
              int need_blocks = n_verts - cnt_block; // how many blocks of current size I still need.
              auto map_blocks = min2(blocks_in_range, need_blocks);
              for (auto l : Range(map_blocks)) {
                for (auto j : Range(block_s)) {
                  d2v_array[range[0]] = cnt_block;
                  v2d_table[cnt_block][bos+j] = range[0]++;
                }
                cnt_block++;
              }
              if (cnt_block == n_verts) {
                bos += block_s;
                block_type++;
                cnt_block = 0;
                if (block_type < O.block_s.Size())
                  { block_s = O.block_s[block_type]; }
              }
            }
          }
        } // range, regular, N dofs, multiple blocks
        break;
      } // REGULAR_ORDERING
      case(Options::DOF_ORDERING::P2_ORDERING):
      {
        assert(strict_alg_mode == false); // "P2_ORDERING with strict alg-mode!"

        this->use_v2d_tab = false;
        this->use_p2_emb = true;

        // cout << " P2-ordering " << endl;
        // cout << " ndof " << ndof << endl;
        // cout << " NV " << ma->GetNV() << endl;

        // cout << bfa << endl;
        // cout << bfa->GetFESpace() << endl;

        // max. #v == fes-#DOFs
        // TODO: throw an error if we get here in strict alg mode
        auto const &fes = *bfa->GetFESpace();

        d2v_array.SetSize(ndof); d2v_array = -1;
        v2d_array.SetSize(ma->GetNV());

        // this->has_node_dofs[NT_EDGE] = false; // this lives in options...

        // !! multidim is assumed right now ~~
        n_verts = 0;
        Array<int> nodeDofs;
        for (auto k : Range(ma->GetNV()))
        {
          fes.GetDofNrs(NodeId(NT_VERTEX, k), nodeDofs);

          if ( nodeDofs.Size() ) // e.g. definedon
          {
            auto const vNum = n_verts++;
            d2v_array[nodeDofs[0]] = vNum;
            v2d_array[vNum] = vNum;
            // cout << " MA-V " << k << " -> DOF " << nodeDofs[0] << " <-> V " << vNum << endl;
          }
        }

        // cout << " FINAL NV " << n_verts << endl;

        v2d_array.SetSize(n_verts);

        // should check for order >= 2, and nodalp2 here !?
        // cout << " ma->GetNEdges() = " << ma->GetNEdges() << std::endl;
        edgePointParents.SetSize(ma->GetNEdges());
        int cntEdgeVs = 0;
        for (auto k : Range(ma->GetNEdges()))
        {
          fes.GetDofNrs(NodeId(NT_EDGE, k), nodeDofs);

          if ( nodeDofs.Size() ) // e.g. definedon
          {
            auto loDOF = nodeDofs[0];
            auto pNums  = ma->GetEdgePNums(k);

            fes.GetDofNrs(NodeId(NT_VERTEX, pNums[0]), nodeDofs);

            if ( nodeDofs.Size() ) // e.g. definedon
            {
              auto d0 = nodeDofs[0];

              if ( nodeDofs.Size() ) // e.g. definedon
              {
                // cout << " edge " << k << ", e-dof " << loDOF << " -> parents (dof " << d0 << " v " << d2v_array[d0] << ") and (dof " << d0 << " v " << d2v_array[d0] << ")" << endl;
                fes.GetDofNrs(NodeId(NT_VERTEX, pNums[1]), nodeDofs);
                auto d1 = nodeDofs[0];

                edgePointParents[cntEdgeVs++] = IVec<3>({loDOF, d2v_array[d0], d2v_array[d1]});
              }
            }
          }
        }
        // cout << "   # MID_SIDE_V = " << cntEdgeVs << endl;
        edgePointParents.SetSize(cntEdgeVs);
        break;
      } // P2_ORDERING
      case(Options::DOF_ORDERING::P2_ORDERING_ALG):
      {
        assert(strict_alg_mode == false); // "P2_ORDERING_ALG without strict alg-mode!");

        this->use_v2d_tab = false;
        this->use_p2_emb = true;

        n_verts = 0;

        // map all edges that do not appear as MSVs to actual vertices
        BitArray isP2Vert(in_ss);
        isP2Vert.Clear();

        for (auto const &trip : algP2Trips)
        {
          // v0, v1, p2-vert
          isP2Vert.SetBit(trip.vMid);
        }

        n_verts = in_ss - isP2Vert.NumSet();

        // cout << " P2_ORDERING_ALG " << endl;
        // cout << " in_ss = " << in_ss << endl;
        // cout << " trips " << algP2Trips.Size() << " -> " << isP2Vert.NumSet() << endl;
        // cout << "   -> " << n_verts << " proper verts left! " << endl;

        d2v_array.SetSize(in_ss); d2v_array = -1;
        v2d_array.SetSize(n_verts);

        n_verts = 0;
        for (auto k : Range(in_ss))
        {
          if ( !isP2Vert.Test(k) )
          {
            auto const vNum = n_verts++;
            v2d_array[vNum] = k;
            d2v_array[k]    = vNum;
          }
        }

        // cout << "   -> " << n_verts << " proper verts CHECK! " << endl;

        edgePointParents.SetSize(algP2Trips.Size());

        for (auto k : Range(algP2Trips))
        {
          auto const &trip = algP2Trips[k];

          // cout << " "
          edgePointParents[k] = IVec<3>{int(trip.vMid), d2v_array[trip.vI], d2v_array[trip.vJ]};
        }
        break;
      }
    }
    break;
  } // RANGE_SUBSET
  case(Options::DOF_SUBSET::SELECTED_SUBSET): {
    // cout << " ss_sel " << O.ss_select << endl;
    const auto & subset = *O.ss_select;
    size_t in_ss = subset.NumSet();

    switch(O.dof_ordering)
    {
      case(VertexAMGPCOptions::DOF_ORDERING::VARIABLE_ORDERING):
      {
        throw Exception("VARIABLE_ORDERING not implemented (how did we get here anyways?)");
        break;
      }
      case(VertexAMGPCOptions::DOF_ORDERING::REGULAR_ORDERING):
      {
        const size_t dpv = std::accumulate(O.block_s.begin(), O.block_s.end(), 0);
        n_verts = in_ss / dpv;

        d2v_array.SetSize(ndof); d2v_array = -1;

        auto n_block_types = O.block_s.Size();

        if (dpv == 1) { // select subset, regular order, 1 dof per V
          v2d_array.SetSize(n_verts);
          auto& subset = *O.ss_select;
          for (int j = 0, k = 0; k < n_verts; j++) {
            // cout << j << " " << k << " " << n_verts << " ss " << subset.Test(j) << endl;
            if (subset.Test(j)) {
              auto d = j; auto svnr = k++;
              d2v_array[d] = svnr;
              v2d_array[svnr] = d;
            }
          }
        }
        else { // select subset, regular order, N dofs per V
          use_v2d_tab = true;
          v2d_table = Table<int>(n_verts, dpv);
          int block_type = 0; // we currently mapping DOFs in O.block_s[block_type]-blocks
          int cnt_block = 0; // how many of those blocks have we gone through
          int block_s = O.block_s[block_type];
          int j = 0, col_os = 0;
          const auto blockss = O.block_s.Size();
          for (auto k : Range(subset.Size())) {
            if (subset.Test(k)) {
              d2v_array[k] = cnt_block;
              v2d_table[cnt_block][col_os + j++] = k;
              if (j == block_s) {
          j = 0;
          cnt_block++;
              }
              if (cnt_block == n_verts) {
          block_type++;
          cnt_block = 0;
          col_os += block_s;
          if (block_type + 1 < blockss)
            { block_s = O.block_s[block_type]; }
              }
            }
          }
        } // select subset, reg. order, N dofs per V

        break;
      }
      case(Options::DOF_ORDERING::P2_ORDERING_ALG):
      case(Options::DOF_ORDERING::P2_ORDERING):
      {
        throw Exception("P2-ordering needs RANGE_SUBSET !?");
        break;
      }
    }
  } // SELECTED_SUBSET
  } // switch(O.subset)

  if (O.vertexSpecification == VertexAMGPCOptions::VERT_SPEC::FROM_MESH_NODES)
  {
    if (inStrictAlgMode())
      { throw Exception("vertexSpecification set to FROM_MESH_NODES in strict alg mode!"); }

    auto fes = bfa->GetFESpace();
    size_t numset = 0;
    O.v_nodes.SetSize(n_verts);

    for (NODE_TYPE NT : { NT_VERTEX, NT_EDGE, NT_FACE, NT_CELL } ) {
      if (numset < n_verts) {
        Array<int> dnums;
        for (auto k : Range(ma->GetNNodes(NT))) {
          NodeId id(NT, k);
          fes->GetDofNrs(id, dnums);
          for (auto dof : dnums) {
            if (IsRegularDof(dof)) { // compressed space
              auto top_vnum = d2v_array[dof];
              if (top_vnum != -1) {
                O.v_nodes[top_vnum] = id;
                numset++;
                break;
              }
            }
          }
        }
        // cout << " after NT " << NT << ", set " << numset << " of " << n_verts << endl;
      }
    }

  }

  // cout << "    DOF <-> AMG-VERTEX mappings:" << endl;
  // cout << " (unsorted) d2v_array: " << endl; prow2(d2v_array); cout << endl << endl;
  // if (use_v2d_tab) {
  //  cout << " (unsorted) v2d_table: " << endl << v2d_table << endl << endl;
  // }
  // else {
  //  cout << " (unsorted) v2d_array: " << endl; prow2(v2d_array); cout << endl << endl;
  // }
} // VertexAMGPC::SetUpMaps


template<class FACTORY>
FACTORY&
VertexAMGPC<FACTORY>::
GetFactory () const
{
  if (_factory == nullptr)
  {
    auto opts = dynamic_pointer_cast<Options>(options);
   _factory =  make_shared<FACTORY>(opts);
  }
  return *_factory;
} // VertexAMGPC::GetFactory


template<class FACTORY>
shared_ptr<BaseDOFMapStep>
VertexAMGPC<FACTORY>::
BuildEmbedding (BaseAMGFactory::AMGLevel & level)
{
  shared_ptr<TopologicMesh> mesh = level.cap->mesh;

  static Timer t("BuildEmbedding"); RegionTimer rt(t);
  auto & O(static_cast<Options&>(*options));

  shared_ptr<BaseMatrix> f_loc_mat;
  if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
    { f_loc_mat = parmat->GetMatrix(); }
  else
    { f_loc_mat = finest_mat; }

  shared_ptr<BaseDOFMapStep> E = nullptr;

  constexpr int max_switch = min(MAX_SYS_DIM, FACTORY::BS);
  Switch<max_switch>(GetEntryDim(f_loc_mat.get())-1, [&, this] (auto BSM) {
    constexpr int BS = BSM + 1;
    if constexpr( (BS == 1) || (BS == 2) || (BS == 3) || (BS == 6) )
      { E = this->BuildEmbedding_impl<BS>(*level.cap); }
    else
      { return; }
  });

  return E;
} // VertexAMGPC::BuildEmbedding


template<class FACTORY>
BaseAMGPC::Options::SM_TYPE
VertexAMGPC<FACTORY>::
SelectSmoother(BaseAMGFactory::AMGLevel const &aMGLevel) const
{
  auto & O (*options);

  auto const smType = O.sm_type.GetOpt(aMGLevel.level);

  switch(smType)
  {
    case(Options::SM_TYPE::GS): { return Options::SM_TYPE::GS; break; }
    case(Options::SM_TYPE::BGS):
    {
      // BGS is only supported on levels where we have the coarse map!
      int no_cmp = (aMGLevel.crs_map == nullptr) ? 1 : 0;

      shared_ptr<ParallelDofs> pardofs = (aMGLevel.level == 0) ? finest_mat->GetParallelDofs()
                                                               : aMGLevel.cap->uDofs.GetParallelDofs();

      if (pardofs != nullptr)
        { pardofs->GetCommunicator().AllReduce(no_cmp, NG_MPI_SUM); }

      return (no_cmp == 0) ? Options::SM_TYPE::BGS
                           : Options::SM_TYPE::GS;
    }
    case(Options::SM_TYPE::JACOBI): { return Options::SM_TYPE::JACOBI; break; }
    case(Options::SM_TYPE::DYNBGS):
    {
      // DYNBGS is probably only useful for special cases (stokes)
      // and also only implemented for scalar matrices
      return Options::SM_TYPE::GS;
      break;
    }
    default:
    {
      // the fancier smoothers introduced for for Stokes are not supported
      // (or needed), so quietly replace with GS
      return Options::SM_TYPE::GS;
      break;
    }
  }
} // BaseAMGPC::SelectSmoother


// template<class FACTORY> template<int BSA>
// tuple<
//   shared_ptr<
//     stripped_spm_tm<
//       Mat<VertexAMGPC<FACTORY>::template BSC<BSA>(),
//           FACTORY::BS,
//           double
//           >
//       >
//     >,
//   UniversalDofs
// >
// VertexAMGPC<FACTORY> :: BuildEDC (size_t height, shared_ptr<TopologicMesh> mesh)
// {
//   return make_tuple(nullptr, UniversalDofs);
// }

template<class FACTORY> template<int BSA>
shared_ptr<BaseDOFMapStep> VertexAMGPC<FACTORY> :: BuildEmbedding_impl (BaseAMGFactory::LevelCapsule const &cap)
{
  auto & O(static_cast<Options&>(*options));

  /**
     Embedding  = E_S * E_D * P
      E_S      ... from 0..ndof to subset                              // entries: N x N
      E_D      ... disp to disp-rot emb or compound-to multidim, etc   // entries: N x dofpv
      P        ... permutation matrix from re-sorting vertex numbers   // entries: dofpv x dofpv
  **/

  auto const &mesh = cap.mesh;

  constexpr int BS = FACTORY::BS;

  shared_ptr<SparseMat<BSA, BSA>> E_S;
  shared_ptr<SparseMat<BSA, BS>>  E_D;
  shared_ptr<SparseMat<BS, BS>>   P;
  shared_ptr<SparseMat<BSA, BS>>  E;

  // shared_ptr<ParallelDofs> fpds = finest_mat->GetParallelDofs();
  UniversalDofs fDofs = MatToUniversalDofs(*finest_mat, DOF_SPACE::ROWS);

  // cout << " fDOFS " << fDofs.GetND() << " " << fDofs.GetNDGlob() << " " << fDofs.GetBS() << endl;

  shared_ptr<BaseMatrix> f_loc_mat = GetLocalMat(finest_mat);

  /** Subset **/
  E_S = BuildES<BSA>();

  /** DOFs **/
  // size_t subset_count = (E_S == nullptr) ? fpds->GetNDofLocal() : E_S->Width();
  size_t subset_count = (E_S == nullptr) ? f_loc_mat->Width() : E_S->Width();
  E_D = BuildED<BSA>(subset_count, mesh);

  /** Permutation **/
  if ( fDofs.IsTrulyParallel() )
  {
    auto & vsort = node_sort[NT_VERTEX];
    P = BuildPermutationMatrix<BS>(vsort);
  }

  // E = E_S * E_D * P
  if ( E_D != nullptr )
  {
    shared_ptr<SparseMat<BSA, BS>> ESED = (E_S == nullptr) ? E_D : MatMultAB(*E_S, *E_D);

    E = (P == nullptr) ? ESED : MatMultAB(*ESED, *P);
  }
  else
  {
    if constexpr(BSA == BS)
    {
      E = (P == nullptr) ? E_S
                         : (E_S == nullptr) ? P
                                            : MatMultAB(*E_S, *P);
    }
    else
    {
      throw Exception("E_D must not be nullptr here!!");
    }
  }

  /** DOF-Map  **/
  shared_ptr<BaseDOFMapStep> emb_step      = nullptr;
  shared_ptr<BaseDOFMapStep> emb_step_comp = nullptr;

  shared_ptr<ParallelDofs> mpds = nullptr;

  /** Zero out rows of embedding for partial Dirichlet **/
  if constexpr(BS * BSA > 1)
  {
    auto scalFree = O.scalFreeRows;

    if (scalFree != nullptr)
    {
      auto const N = subset_count;

      if (E == nullptr)
      {
        Array<int> perow(N);
        perow = 1;
        E = make_shared<SparseMat<BSA, BS>>(perow, N);
        for (auto k : Range(N))
        {
          E->GetRowIndices(k)[0] = k;
          SetIdentity(E->GetRowValues(k)[0]);
        }
      }

      for (auto k : Range(N))
      {
        Iterate<BSA>([&](auto l)
        {
          auto scalRow = BSA * k + l;

          if (!scalFree->Test(scalRow))
          {
            auto rvs = E->GetRowValues(k);

            for (auto j : Range(rvs))
            {
              Iterate<BS>([&](auto ll) {
                rvs[j](l, ll) = 0.0;
              });
            }
          }
        });
      }
    }
  }

  int have_embed = (E == nullptr) ? 0 : 1;

  if (fDofs.GetCommunicator().Size() > 1)
    { have_embed = fDofs.GetCommunicator().AllReduce(have_embed, NG_MPI_SUM); }

  if (have_embed) {

    UniversalDofs meshDofs = GetFactory().BuildUDofs(cap);

    if ( E == nullptr ) {
      // probably just for rank 0! (ParallelDofs - constructor has to be called by every member of the communicator!)
      Array<int> perow(fDofs.GetND()); perow = 1;
      E   = make_shared<SparseMat<BSA, BS>>(perow, meshDofs.GetND());

      // if constexpr(false)
      //   { E_C = make_shared<T_E_D_C>(perow, meshDofs.GetND()); }

      for (auto k : Range(perow.Size()))
      {
        E->GetRowIndices(k)[0] = k;
        SetIdentity(E->GetRowValues(k)[0]);

        // if constexpr(false)
        // {
        //   E_C->GetRowIndices(k)[0] = k;
        //   SetIdentity(E_C->GetRowValues(k)[0]);
        // }
      }
    }

    if (O.log_level_pc == Options::LOG_LEVEL_PC::DBG)
    {
      int const rk = fDofs.GetCommunicator().Rank();
      ofstream out("embed_map_rk" + std::to_string(rk) + ".out");
      out << *E << endl;
    }

    emb_step = make_shared<ProlMap<StripTM<BSA, BS>>>(E,
                                                      fDofs,
                                                      meshDofs);

    // if constexpr(true)
    // {
      // emb_step = make_shared<ProlMap<SparseMatTM<BSA, BS>>>(E, fDofs, meshDofs);
    // }
    // else
    // {
    //   // for debugging: rot-comp of prol-BFs, only for debugging
    //   Array<shared_ptr<BaseDOFMapStep>> sub_steps(2);
    //   sub_steps[0] = make_shared<ProlMap<T_E_D>>(E, fDofs, meshDofs);

    //   UniversalDofs fDofs_C(E_C->Height(), BSC<BSA>());

    //   sub_steps[1] = make_shared<ProlMap<T_E_D_C>>(E_C, fDofs_C, meshDofs);

    //   emb_step = make_shared<MultiDofMapStep>(sub_steps);
    // }
  }

  return emb_step;
} // VertexAMGPC::BuildEmbedding_impl


template<class FACTORY>
template<int BSA>
shared_ptr<SparseMat<BSA, BSA>>
VertexAMGPC<FACTORY>::
BuildES ()
{
  const auto & O(static_cast<Options&>(*options));

  shared_ptr<ParallelDofs> fpds = finest_mat->GetParallelDofs();
  // size_t ndLocal = (fpds == nullptr) ? finest_mat->Height() : fpds->GetNDofLocal();
  size_t nd_loc = finest_mat->Height();

  typedef SparseMat<BSA, BSA> TS;
  shared_ptr<TS> E_S = nullptr;
  switch(O.subset)
  {
    case(Options::DOF_SUBSET::RANGE_SUBSET):
    {
      IVec<2, size_t> notin_ss = {0, nd_loc };
      for (auto pair : O.ss_ranges)
      {
        if (notin_ss[0] == pair[0])
          { notin_ss[0] = pair[1]; }
        else if (notin_ss[1] == pair[1])
          { notin_ss[1] = pair[0]; }
      }

      int is_triv = ( (notin_ss[1] - notin_ss[0]) == 0 ) ? 1 : 0;

      if (fpds != nullptr)
        { fpds->GetCommunicator().AllReduce(is_triv, NG_MPI_SUM); }

      if (is_triv == 0)
      {
        Array<int> perow(nd_loc); perow = 0;
        int cnt_cols = 0;

        for (auto pair : O.ss_ranges)
          if (pair[1] > pair[0])
            { perow.Range(pair[0], pair[1]) = 1; cnt_cols += pair[1] - pair[0]; }

        E_S = make_shared<TS>(perow, cnt_cols); cnt_cols = 0;

        for (auto pair : O.ss_ranges)
        {
          for (auto c : Range(pair[0], pair[1]))
          {
            SetIdentity(E_S->GetRowValues(c)[0]);
            E_S->GetRowIndices(c)[0] = cnt_cols++;
          }
        }
      }
      break;
    } // case(Options::DOF_SUBSET::RANGE_SUBSET):
    case(Options::DOF_SUBSET::SELECTED_SUBSET):
    {
      if (O.ss_select == nullptr)
        { throw Exception("SELECTED_SUBSET, but no ss_select!"); }

      const auto & SS(*O.ss_select);
      int is_triv = (SS.NumSet() == SS.Size()) ? 1 : 0;

      if (fpds != nullptr)
        { fpds->GetCommunicator().AllReduce(is_triv, NG_MPI_SUM); }

      if (is_triv == 0)
      {
        int cnt_cols = SS.NumSet();
        Array<int> perow(nd_loc);
        for (auto k : Range(perow))
          { perow[k] = SS.Test(k) ? 1 : 0; }

        E_S = make_shared<TS>(perow, cnt_cols); cnt_cols = 0;

        for (auto k : Range(nd_loc))
        {
          if (SS.Test(k))
          {
            SetIdentity(E_S->GetRowValues(k)[0]);
            E_S->GetRowIndices(k)[0] = cnt_cols++;
          }
        }
      }
      break;
    } // case(Options::DOF_SUBSET::SELECTED_SUBSET)
  } // switch(O.subset)

  return E_S;
} // VertexAMGPC::BuildES


template<class FACTORY>
shared_ptr<BlockTM> VertexAMGPC<FACTORY> :: BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h)
{
  Options & O = static_cast<Options&>(*options);

  shared_ptr<BlockTM> top_mesh;
  switch(O.topo) {
  case(Options::TOPO::MESH_TOPO):  { top_mesh = BTM_Mesh(eqc_h); break; }
  case(Options::TOPO::ALG_TOPO):   { top_mesh = BTM_Alg(eqc_h); break; }
  case(Options::TOPO::ELMAT_TOPO): { throw Exception("cannot do topology from elmats!"); break; }
  default: { throw Exception("invalid topology type!"); break; }
  }

  return top_mesh;
} // VertexAMGPC::BuildTopMesh


template<class FACTORY>
shared_ptr<BlockTM> VertexAMGPC<FACTORY> :: BTM_Mesh (shared_ptr<EQCHierarchy> eqc_h)
{
  throw Exception("I don't think this is functional...");

  // const auto &O(*options);
  // node_sort.SetSize(4);
  // shared_ptr<BlockTM> top_mesh;
  // switch (O.v_pos) {
  // case(BAO::VERTEX_POS): {
  //   top_mesh = MeshAccessToBTM (ma, eqc_h, node_sort[0], true, node_sort[1],
  // 				  false, node_sort[2], false, node_sort[3]);
  //   break;
  // }
  // case(BAO::GIVEN_POS): {
  //   throw Exception("Cannot combine custom vertices with topology from mesh");
  //   break;
  // }
  // default: { throw Exception("kinda unexpected case"); break; }
  // }
  // // TODO: re-map d2v/v2d
  // // this only works for the simplest case anyways...
  // auto fvs = make_shared<BitArray>(top_mesh->template GetNN<NT_VERTEX>()); fvs->Clear();
  // auto & vsort = node_sort[NT_VERTEX];
  // for (auto k : Range(top_mesh->template GetNN<NT_VERTEX>()))
  //   if (finest_freedofs->Test(k))
  // 	{ fvs->SetBit(vsort[k]); }
  // free_verts = fvs;
  // return top_mesh;

  return nullptr;
} // VertexAMGPC::BTM_Mesh


template<class FACTORY>
shared_ptr<BlockTM> VertexAMGPC<FACTORY> :: BTM_Alg (shared_ptr<EQCHierarchy> eqc_h)
{
  static Timer t("BTM_Alg"); RegionTimer rt(t);

  auto & O = static_cast<Options&>(*options);
  node_sort.SetSize(4);

  // cout << " eqc_h = " << *eqc_h << endl;

  auto top_mesh = make_shared<BlockTM>(eqc_h);

  size_t n_verts = 0, n_edges = 0;

  auto & vert_sort = node_sort[NT_VERTEX];

  auto fpd = finest_mat->GetParallelDofs();

  /** Vertices **/
  auto set_vs = [&](auto nv, auto v2d) {
    n_verts = nv;

    // if fpd is nullptr, eqc_h is dummy and this will not be called
    auto dpofv = [&](auto vnr) LAMBDA_INLINE { return fpd->GetDistantProcs(v2d(vnr)); };

    vert_sort.SetSize(nv);
    auto sortv = [&vert_sort](auto i, auto j) LAMBDA_INLINE { vert_sort[i] = j; };

    top_mesh->SetVs (nv, dpofv, sortv);

    free_verts = make_shared<BitArray>(nv);

    if (finest_freedofs != nullptr) {
      // cout << " finest_freedofs 1: " << finest_freedofs << endl;
      // if (finest_freedofs)
        // { prow2(*finest_freedofs); cout << endl; }
      free_verts->Clear();
      for (auto k : Range(nv)) {
        // cout << k << " dof " << v2d(k) << " sort " << vert_sort[k] << " free " << finest_freedofs->Test(v2d(k)) << endl;
        if (finest_freedofs->Test(v2d(k)))
          { free_verts->SetBit(vert_sort[k]); }
      }
    }
    else
      { free_verts->Set(); }
    // cout << "diri verts: " << endl;
    // for (auto k : Range(free_verts->Size()))
    // 	if (!free_verts->Test(k)) { cout << k << " " << endl; }
    // cout << endl;
    // cout << "diri DOFs: " << endl;
    // for (auto k : Range(finest_freedofs->Size()))
    // 	if (!finest_freedofs->Test(k)) { cout << k << " " << endl; }
    // cout << endl;
    // cout << " VSORT: " << endl; prow2(vert_sort); cout << endl;
  };

  if (use_v2d_tab) {
    set_vs(v2d_table.Size(), [&](auto i) LAMBDA_INLINE { return v2d_table[i][0]; });
  }
  else {
    set_vs(v2d_array.Size(), [&](auto i) LAMBDA_INLINE { return v2d_array[i]; });
  }

  /** Edges **/
  auto create_edges = [&](auto v2d, auto d2v) LAMBDA_INLINE {

    auto traverse_graph = [&](const auto& g, auto fun) LAMBDA_INLINE { // vertex->dof,  // dof-> vertex
      for (auto k : Range(n_verts)) {
        int row = v2d(k); // for find_in_sorted_array
        auto ri = g.GetRowIndices(row);
        auto pos = find_in_sorted_array(row, ri); // no duplicates
        if (pos+1 < ri.Size()) {
          for (auto col : ri.Part(pos+1)) {
            auto j = d2v(col);
            if (j != -1)
              { fun(vert_sort[k],vert_sort[j]); }
          }
        }
      }
    }; // traverse_graph

    auto bspm = dynamic_pointer_cast<BaseSparseMatrix>(finest_mat);
    if (!bspm)
      { bspm = dynamic_pointer_cast<BaseSparseMatrix>( dynamic_pointer_cast<ParallelMatrix>(finest_mat)->GetMatrix()); }
    if (!bspm)
      { throw Exception("could not get BaseSparseMatrix out of finest_mat!!"); }

    n_edges = 0;
    traverse_graph(*bspm, [&](auto vk, auto vj) LAMBDA_INLINE { n_edges++; });

    Array<decltype(AMG_Node<NT_EDGE>::v)> epairs(n_edges);

    n_edges = 0;
    traverse_graph(*bspm, [&](auto vk, auto vj) LAMBDA_INLINE {
      if (vk < vj) { epairs[n_edges++] = { vk, vj }; }
      else { epairs[n_edges++] = { vj, vk }; }
    });

    // cout << " edge pair list: " << endl;
    // prow2(epairs); cout << endl;
    top_mesh->SetNodes<NT_EDGE> (n_edges, [&](auto num) LAMBDA_INLINE { return epairs[num]; }, // (already v-sorted)
                                          [](auto node_num, auto id) LAMBDA_INLINE { /* dont care about edge-sort! */ });

    auto tme = top_mesh->GetNodes<NT_EDGE>();
    // cout << " tmesh edges: " << endl << tme << endl;
    // cout << "final n_edges: " << top_mesh->GetNN<NT_EDGE>() << endl;
  }; // create_edges

  // auto create_edges = [&](auto v2d, auto d2v) LAMBDA_INLINE {
  if (use_v2d_tab) {
    // create_edges([&](auto i) LAMBDA_INLINE { return v2d_table[i][0]; },
    // 		   [&](auto i) LAMBDA_INLINE { return d2v_array[i]; } );
    create_edges([&](auto i) LAMBDA_INLINE { return v2d_table[i][0]; },
      [&](auto i) LAMBDA_INLINE { // I dont like it ...
        auto v = d2v_array[i];
        if ( (v != -1) && (v2d_table[v][0] == i) )
          { return v; }
        return -1;
      });
  }
  else {
    create_edges([&](auto i) LAMBDA_INLINE { return v2d_array[i]; },
      [&](auto i) LAMBDA_INLINE { return d2v_array[i]; } );
  }

  // update v2d/d2v with vert_sort
  if (use_v2d_tab) {
    Array<int> cnt(n_verts); cnt = 0;
    for (auto k : Range(n_verts)) {
      auto vk = vert_sort[k];
      for (auto d : v2d_table[k])
        { d2v_array[d] = vk; }
    }
    for (auto k : Range(d2v_array)) {
      auto vnr = d2v_array[k];
      if  (vnr != -1)
        { v2d_table[vnr][cnt[vnr]++] = k; }
    }
  }
  else {
    for (auto k : Range(n_verts)) {
      auto d = v2d_array[k];
      d2v_array[d] = vert_sort[k];
    }
    for (auto k : Range(d2v_array))
      if  (d2v_array[k] != -1)
        { v2d_array[d2v_array[k]] = k; }
  }

  //cout << " (sorted) d2v_array: " << endl; prow2(d2v_array); cout << endl << endl;
  //if (use_v2d_tab) {
    //cout << " (sorted) v2d_table: " << endl << v2d_table << endl << endl;
  //}
  //else {
  // cout << " (sorted) v2d_array: " << endl; prow2(v2d_array); cout << endl << endl;
  //}

  //cout << " FINEST TOP MESH " << endl << *top_mesh << endl;

  // std::ofstream os("finest_top_mesh.out");
  // os << *top_mesh << endl;

  return top_mesh;
} // VertexAMGPC::BTM_Alg




template<class FACTORY>
shared_ptr<typename FACTORY::TMESH> VertexAMGPC<FACTORY> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
{
  Options & O = static_cast<Options&>(*options);

  shared_ptr<TMESH> alg_mesh;

  switch(O.energy) {
  case(Options::TRIV_ENERGY): { alg_mesh = BuildAlgMesh_TRIV(top_mesh); break; }
  case(Options::ALG_ENERGY): { alg_mesh = BuildAlgMesh_ALG(top_mesh); break; }
  case(Options::ELMAT_ENERGY): { alg_mesh = BuildAlgMesh_ELMAT(top_mesh); break;  }
  default: { throw Exception("Invalid Energy!"); break; }
  }

  return alg_mesh;
} // VertexAMGPC::BuildAlgMesh


template<class FACTORY>
shared_ptr<typename FACTORY::TMESH> VertexAMGPC<FACTORY> :: BuildAlgMesh_ELMAT(shared_ptr<BlockTM> top_mesh)
{
  throw Exception("ELMAT energy for VertexAMGPC, how did we get here?");
  return nullptr;
}

template<class FACTORY>
shared_ptr<typename FACTORY::TMESH> VertexAMGPC<FACTORY> :: BuildAlgMesh_ALG (shared_ptr<BlockTM> top_mesh)
{
  static Timer t("BuildAlgMesh_ALG"); RegionTimer rt(t);

  shared_ptr<typename FACTORY::TMESH> alg_mesh;

  shared_ptr<BaseMatrix> f_loc_mat;
  if (auto parmat = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
    { f_loc_mat = parmat->GetMatrix(); }
  else
    { f_loc_mat = finest_mat; }
  auto spmat = dynamic_pointer_cast<BaseSparseMatrix>(f_loc_mat);

  if (use_v2d_tab) {
    alg_mesh = BuildAlgMesh_ALG_blk(top_mesh, spmat,
            [&](auto d) LAMBDA_INLINE { return d2v_array[d]; },
            [&](auto v) LAMBDA_INLINE { return v2d_table[v]; } );
  }
  else {
    alg_mesh = BuildAlgMesh_ALG_scal(top_mesh, spmat,
              [&](auto d) LAMBDA_INLINE { return d2v_array[d]; },
              [&](auto v) LAMBDA_INLINE { return v2d_array[v]; } );
  }

  return alg_mesh;
} // EmbedVAMG::BuildAlgMesh_ALG


template<class FACTORY>
void VertexAMGPC<FACTORY> :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
{
  BaseAMGPC::InitFinestLevel(finest_level);
  /** Explicitely assemble matrix associated with the finest mesh. **/
  finest_level.cap->free_nodes = free_verts;
}


template<class FACTORY>
Table<int> VertexAMGPC<FACTORY> :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
{
  if (amg_level.crs_map == nullptr) {
    throw Exception("Crs Map not saved!!");
    return std::move(Table<int>());
  }

  auto & O(static_cast<Options&>(*options));

  // if (amg_level.level == 0 && O.smooth_lo_only) {
  //   throw Exception("Asked for BGS blocks for HO part!!");
  //   return std::move(Table<int>());
  // }

  int NCV = amg_level.crs_map->GetMappedNN<NT_VERTEX>();
  int n_blocks = NCV;
  // if (amg_level.disc_map != nullptr)
    // { n_blocks += amg_level.disc_map->GetNDroppedNodes<NT_VERTEX>(); }
  TableCreator<int> cblocks(n_blocks);
  auto it_blocks = [&](auto NV, auto map_v) LAMBDA_INLINE {
    for (auto k : Range(NV)) {
      auto cv = map_v(k);
      if (cv != -1)
        { cblocks.Add(cv, k); }
    }
  };
  auto it_blocks_2 = [&](auto NV, auto map_v) LAMBDA_INLINE {
    if (use_v2d_tab) {
      for (auto k : Range(NV)) {
        auto cv = map_v(k);
        if (cv != -1)
          for (auto dof : v2d_table[k])
            { cblocks.Add(cv, dof); }
      }
    }
    else {
      for (auto k : Range(NV)) {
        auto cv = map_v(k);
        if (cv != -1)
          { cblocks.Add(cv, v2d_array[k]); }
      }
    }
  };
  auto vmap = amg_level.crs_map->GetMap<NT_VERTEX>();

  // TODO: clean BGS-blocks for pre/post L0 emb, enable BGS for pre-emb when smooth_after_emb (somehow?)
  //                add something like pre_emb_sm_type option, add a "GetPreEmbBlocks", route getgsblocks there
  //                if not smooth_after_emb

  for (; !cblocks.Done(); cblocks++) {
    if (true) { // (amg_level.disc_map == nullptr) {
      auto map_v = [&](auto v) -> int LAMBDA_INLINE { return vmap[v]; };
      if (amg_level.level == 0 && (!O.smooth_after_emb)) // if smooth_after_emb, need mesh-canonic blocks
        { it_blocks_2(vmap.Size(), map_v); }
      else
        { it_blocks(vmap.Size(), map_v); }
    }
    else {
      // const auto & drop = *amg_level.disc_map->GetDroppedNodes<NT_VERTEX>();
      // auto drop_map = amg_level.disc_map->GetMap<NT_VERTEX>();
      // auto map_v =  [&](auto v) -> int LAMBDA_INLINE {
      //   auto midv = drop_map[v]; // have to consider drop OR DIRI !!
      //   return (midv == -1) ? midv : vmap[midv];
      // };
      // if (amg_level.level == 0) {
      //   it_blocks_2(drop.Size(), map_v);
      //   int c = NCV;
      //   if (use_v2d_tab) {
      //     for (auto k : Range(drop.Size())) {
      //       if (drop.Test(k)) {
      //         for (auto dof : v2d_table[k])
      //           { cblocks.Add(c, dof); }
      //         c++;
      //       }
      //     }
      //   }
      //   else {
      //     for (auto k : Range(drop.Size())) {
      //       if (drop.Test(k))
      //         { cblocks.Add(c++, v2d_array[k]); }
      //     }
      //   }
      // }
      // else {
      //   it_blocks(drop.Size(), map_v);
      //   int c = NCV;
      //   for (auto k : Range(drop.Size())) {
      //     if (drop.Test(k))
      //       { cblocks.Add(c++, k); }
      //   }
      // }
    }
  }

  auto blocks = cblocks.MoveTable();

  return std::move(blocks);
} // VertexAMGPC<FACTORY>::GetGSBlocks


template<class FACTORY>
Table<int>
VertexAMGPC<FACTORY>::
GetFESpaceGSBlocks()
{
  if (strict_alg_mode)
  {
    throw Exception("GetFESpaceGSBlocks called in strict alg mode!");
  }
  
  Flags flags;
  flags.SetFlag("eliminate_internal", bfa->UsesEliminateInternal());

  auto fesBlocks = bfa->GetFESpace()->CreateSmoothingBlocks(flags);

  Table<int> blocks(*fesBlocks);

  return blocks;
}

template<class FACTORY>
IVec<3, double> VertexAMGPC<FACTORY> :: GetElmatEVs() const
{
  return IVec<3,double>({ 0.0, std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity() });
} // VertexAMGPC::GetElmatEVs


template<class FACTORY>
void
VertexAMGPC<FACTORY>::
SetNodalP2Connectivity(FlatArray<AMGSolverSettings::NodalP2Triple> p2Trips)
{
  auto & O = static_cast<Options&>(*options);

  O.subset = Options::DOF_SUBSET::RANGE_SUBSET;
  O.dof_ordering = Options::DOF_ORDERING::P2_ORDERING_ALG;
  this->algP2Trips.Assign(p2Trips);
}


/** END VertexAMGPC **/


/** ElmatVAMG **/


template<class FACTORY, class HTVD, class HTED>
ElmatVAMG<FACTORY, HTVD, HTED> :: ElmatVAMG (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
  : VertexAMGPC<FACTORY>(blf, flags, name, opts), ht_vertex(nullptr), ht_edge(nullptr)
{
  elmat_evs = { std::numeric_limits<double>::max() , 0.0, 0.0 };
} // ElmatVAMG(..)


template<class FACTORY, class HTVD, class HTED>
ElmatVAMG<FACTORY, HTVD, HTED> :: ElmatVAMG (shared_ptr<BaseMatrix> A, Flags const &flags, const string name, shared_ptr<Options> opts)
  : VertexAMGPC<FACTORY>(A, flags, name, opts)
  , ht_vertex(nullptr)
  , ht_edge(nullptr)
{
  ;
}

template<class FACTORY, class HTVD, class HTED>
ElmatVAMG<FACTORY, HTVD, HTED> :: ~ElmatVAMG ()
{
} // ~ElmatVAMG


template<class FACTORY, class HTVD, class HTED>
shared_ptr<BlockTM> ElmatVAMG<FACTORY, HTVD, HTED> :: BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h)
{
  Options & O = static_cast<Options&>(*options);

  shared_ptr<BlockTM> top_mesh;
  switch(O.topo) {
    case(Options::TOPO::MESH_TOPO):  { top_mesh = BTM_Mesh(eqc_h); break; }
    case(Options::TOPO::ALG_TOPO):   { top_mesh = BTM_Alg(eqc_h); break; }
    case(Options::TOPO::ELMAT_TOPO): { top_mesh = BTM_Elmat(eqc_h); break; }
    default: { throw Exception("invalid topology type!"); break; }
  }

  return top_mesh;
} // ElmatVAMG::BuildTopMesh


template<class FACTORY, class HTVD, class HTED>
shared_ptr<BlockTM> ElmatVAMG<FACTORY, HTVD, HTED> :: BTM_Elmat (shared_ptr<EQCHierarchy> eqc_h)
{
  throw Exception("topologoy from element matrices are a TODO, use one of the other options in the meantime");
  return nullptr;
} // ElmatVAMG::BTM_Elmat

template<class FACTORY, class HTVD, class HTED>
void ElmatVAMG<FACTORY, HTVD, HTED> :: InitLevel (shared_ptr<BitArray> freedofs)
{
  BASE::InitLevel(freedofs);

  Options & O = static_cast<Options&>(*options);

  elmat_evs = { std::numeric_limits<double>::max() , 0.0, 0.0 };

  if (O.energy == Options::ENERGY::ELMAT_ENERGY) {
    //  - overestimates for compound spaces ?!
    //  - probably the entire thing does not work for multidim ?!
    //  - WHY do this at all, can't we just do sth like BTM_Fes
    //    and then only ADD weights in AddElementMatrix??
    shared_ptr<FESpace> lofes = this->bfa->GetFESpace();
    if (auto V = lofes->LowOrderFESpacePtr())
      { lofes = V; }
    size_t NV = lofes->GetNDof();
    ht_vertex = make_unique<HashTable<int, HTVD>>(NV);
    // maybe instead: lofes->GetMeshAccess()->GetNEdges();
    ht_edge = make_unique<HashTable<IVec<2,int>, HTED>>(12 * NV);
  }
} // ElmatVAMG::InitLevel


template<class FACTORY, class HTVD, class HTED>
void ElmatVAMG<FACTORY, HTVD, HTED> :: FinalizeLevel (shared_ptr<BaseMatrix> mat)
{
  Options & O = static_cast<Options&>(*options);

  if (O.calc_elmat_evs)
  {
    std::cout << " worst MIN EV             = " << elmat_evs[0] << std::endl;
    std::cout << " worst MAX EV             = " << elmat_evs[1] << std::endl;
    std::cout << " worst real kappa         = " << elmat_evs[2] << std::endl;
    std::cout << " pessimistic global kappa = " << kappaOf(elmat_evs[0], elmat_evs[1]) << std::endl;
  }

  BASE::FinalizeLevel(mat);
} // ElmatVAMG::FinalizeLevel


template<class FACTORY, class HTVD, class HTED>
IVec<3, double> ElmatVAMG<FACTORY, HTVD, HTED> :: GetElmatEVs() const
{
  return elmat_evs;
} // ElmatVAMG :: GetElmatEVs


/** END ElmatVAMG **/

} // namespace amg

#endif // FILE_AMG_PC_VERTEX_IMPL_HPP
