#ifndef FILE_ELAST_PC_IMPL_HPP
#define FILE_ELAST_PC_IMPL_HPP

#include "base_factory.hpp"
#include "utils_denseLA.hpp"

namespace amg
{

template<class FACTORY, class HTVD, class HTED>
void ElmatVAMG<FACTORY, HTVD, HTED> :: AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
                ElementId ei, LocalHeap & lh)
{
  const auto & O(static_cast<Options&>(*options));

  if (O.energy != Options::ENERGY::ELMAT_ENERGY)
    { return; }

  if (O.aux_elmat_version == 0)
  {
    CalcAuxWeightsSC(dnums, elmat, ei, lh);
  }
  else if (O.aux_elmat_version == 1)
  {
    CalcAuxWeightsALG(dnums, elmat, ei, lh);
  }
  else
  {
    CalcAuxWeightsLSQ(dnums, elmat, ei, lh);
  }

} // ElmatVAMG::AddElementMatrix

template<class FCC>
void VertexAMGPC<FCC> :: SetDefaultOptions (BaseAMGPC::Options& base_O)
{
  auto & O(static_cast<Options&>(base_O));

  shared_ptr<MeshAccess> ma = inStrictAlgMode() ? nullptr : bfa->GetFESpace()->GetMeshAccess();

  /** vertex positions are needed! **/
  O.vertexSpecification = inStrictAlgMode() ?
                            VertexAMGPCOptions::VERT_SPEC::ALGEBRAIC :
                            VertexAMGPCOptions::VERT_SPEC::FROM_MESH_NODES;

  /** Coarsening Algorithm **/
  // O.crs_alg = Options::CRS_ALG::MIS;
  // O.ecw_geom = false;
  // O.mis_dist2 = SpecOpt<bool>(false, { true });
  // O.mis_neib_boost = false; // might be worth it

  O.crs_alg = Options::CRS_ALG::SPW;
  O.ecw_geom = false;
  O.crs_robust = false;

  /** Smoothed Prolongation **/
  O.enable_sp = true;
  O.sp_needs_cmap = false;
  O.sp_min_frac = ( FCC::DIM == 3 ) ? 0.08 : 0.15;
  O.sp_max_per_row = 1 + FCC::DIM;
  O.sp_omega = 1.0;

  // /** Discard **/
  // O.enable_disc = false; // this has always been a hack, so turn it off by default...
  // O.disc_max_bs = 1; // TODO: make this work

  /** Level-control **/
  O.enable_multistep = false;
  O.use_static_crs = true;
  O.first_aaf = ( FCC::DIM == 3 ) ? 0.025 : 0.05;
  O.aaf = 1/pow(2, FCC::DIM);

  /** Redistribute **/
  O.enable_redist = true;
  O.rdaf = 0.05;
  O.first_rdaf = O.aaf * O.first_aaf;
  O.rdaf_scale = 1;
  O.rd_crs_thresh = 0.9;
  O.rd_min_nv_gl = 5000;
  O.rd_seq_nv = 5000;

  /** Smoothers **/
  O.sm_type = Options::SM_TYPE::GS;
  O.keep_grid_maps = false;

  /** make a guess wether we have rotations or not, and how the DOFs are ordered **/
  std::function<Array<size_t>(shared_ptr<FESpace>)> check_space = [&](auto fes) -> Array<size_t> {
    auto fes_dim = fes->GetDimension();
    // cout << " fes " << typeid(*fes).name() << " dim " << fes_dim << endl;
    if (auto comp_fes = dynamic_pointer_cast<CompoundFESpace>(fes)) {
      auto n_spaces = comp_fes->GetNSpaces();
      Array<size_t> comp_bs;
      for (auto k : Range(n_spaces))
        { comp_bs.Append(check_space((*comp_fes)[k])) ; }
      size_t sum_bs = std::accumulate(comp_bs.begin(), comp_bs.end(), 0);
      if (sum_bs == FCC::ENERGY::DPV)
        { O.with_rots = true; }
      else if (sum_bs == FCC::ENERGY::DISPPV)
        { O.with_rots = false; }
      return comp_bs;
    }
    else if (auto reo_fes = dynamic_pointer_cast<ReorderedFESpace>(fes)) {
      // reordering changes [1,1,1] -> [3]
      auto base_space = reo_fes->GetBaseSpace();
      auto unreo_bs = check_space(base_space);
      size_t sum_bs = std::accumulate(unreo_bs.begin(), unreo_bs.end(), 0);
      if (sum_bs == FCC::ENERGY::DPV)
        { O.with_rots = true; }
      else if (sum_bs == FCC::ENERGY::DISPPV)
        { O.with_rots = false; }
      return Array<size_t>({sum_bs});
    }
    else if (fes_dim == FCC::ENERGY::DPV) { // only works because mdim+compound does not work, so we never land here in the compound case
      O.with_rots = true;
      return Array<size_t>({1});
    }
    else if (auto compr_fes = dynamic_pointer_cast<CompressedFESpace>(fes))
      { return check_space(compr_fes->GetBaseSpace()); }
    else if (fes_dim == FCC::ENERGY::DISPPV) {
      O.with_rots = false;
      return Array<size_t>({1});
    }
    else
      { return Array<size_t>({1}); }
  };

  if (inStrictAlgMode())
  {
    // best guess in strict alg mode is that we have a displacement-only formulation,
    // and that we are using sparse matrice swith Mat<DIM,DIM> entries 
    O.block_s          = Array<size_t>({1});
    O.with_rots        = false;
    O.regularize_cmats = true;
  }
  else
  {
    O.block_s = check_space(bfa->GetFESpace());
    O.regularize_cmats = !O.with_rots; // without rotations on finest level, we can get singular coarse mats !
  }

} // VertexAMGPC::SetDefaultOptions


template<class FCC>
void VertexAMGPC<FCC> :: ModifyOptions (BaseAMGPC::Options & aO, const Flags & flags, string prefix)
{
  auto & O(static_cast<Options&>(aO));
  if ( (O.sm_type.default_opt == Options::SM_TYPE::BGS) ||
        (O.sm_type.spec_opt.Pos(Options::SM_TYPE::BGS) != -1) )
    { O.keep_grid_maps = true; }
} // VertexAMGPC::ModifyOptions


template<class FCC>
void
VertexAMGPC<FCC>::
SetVertexCoordinates(FlatArray<double> coords)
{
  auto & O(static_cast<Options&>(*options));

  size_t NV = coords.Size() / FCC::DIM;

  O.algVPositions.SetSize(NV);

  for (auto k : Range(NV))
  {
    size_t offs = k * FCC::DIM;
    Iterate<FCC::DIM>([&](auto l) {
      O.algVPositions[k](l.value) = coords[offs + l.value];
    });
  }
} // VertexAMGPC::SetVertexPositions


template<class FACTORY>
void
VertexAMGPC<FACTORY>::
SetEmbProjScalRows(shared_ptr<BitArray> scalFree)
{
  auto & O(static_cast<Options&>(*options));

  O.scalFreeRows = scalFree;
} // VertexAMGPC::SetEmbProjScalRows


template<int N>
double InnerProduct(Mat<N, N, double> const &A,
                    Mat<N, N, double> const &B)
{
  double ip = 0.0;
  Iterate<N>([&](auto i) {
    Iterate<N>([&](auto j) {
      ip += A(i.value, j.value) * B(i.value, j.value);
    });
  });
  return ip;
}

// returns alpha such that alpha * A ~ B in least squares sense
template<class TM>
double leastSquaresFit(TM const &A,
                       TM const &B)
{
  return InnerProduct(A, B) / InnerProduct(B, B);
}

template<class FCC> template<class TD2V, class TV2D> shared_ptr<typename FCC::TMESH>
VertexAMGPC<FCC> :: BuildAlgMesh_ALG_scal (shared_ptr<BlockTM> top_mesh,
                                            shared_ptr<BaseSparseMatrix> spmat,
                                            TD2V D2V, TV2D V2D) const
{
  static Timer ti("BuildAlgMesh_ALG_scal"); RegionTimer rt(ti);
  const auto & O(static_cast<Options&>(*options));

  const auto& dof_blocks(O.block_s);
  if ( (dof_blocks.Size() != 1) || (dof_blocks[0] != 1) )
    { throw Exception("block_s for compound, but called algmesh_alg_scal!"); }

  /** Vertex Data  **/
  auto a = new AttachedEVD<DIM>(Array<ElastVData<FCC::DIM>>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
  auto vdata = a->Data(); // TODO: get penalty dirichlet from row-sums (only taking x/y/z displacement entries)
  FlatArray<int> vsort = node_sort[NT_VERTEX];
  
  if (O.vertexSpecification == VertexAMGPCOptions::VERT_SPEC::FROM_MESH_NODES)
  {
    const auto & MA(*ma);
    Vec<FCC::DIM> t;
    for (auto k : Range(O.v_nodes)) {
      auto vnum = vsort[k];
      vdata[vnum].wt = 0;
      vdata[vnum].rot_scaling = 1.0;
      GetNodePos(O.v_nodes[k], MA, vdata[vnum].pos, t);
    }
  }
  else
  {
    for (auto vnum : Range(top_mesh->template GetNN<NT_VERTEX>()))
    {
      auto dnr = V2D(vnum); // this includes vsort! (see BTM_Alg)
      vdata[vnum].wt          = 0;
      vdata[vnum].rot_scaling = 1.0;
      vdata[vnum].pos         = O.algVPositions[dnr];
    }
  }


  /** Edge Data  **/
  auto b = new AttachedEED<FCC::DIM>(Array<ElasticityEdgeData<FCC::DIM>>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED); // !! has to be distr
  auto edata = b->Data();
  auto edges = top_mesh->GetNodes<NT_EDGE>();

  Array<double> rot_scalings(top_mesh->GetNN<NT_VERTEX>()); rot_scalings = 0.0;

  if (auto spm_tm = dynamic_pointer_cast<SparseMatrixTM<Mat<FCC::ENERGY::DISPPV,FCC::ENERGY::DISPPV,double>>>(spmat)) { // disp only
    const auto& A(*spm_tm);
    for (auto & e : edges) {
      auto di = V2D(e.v[0]); auto dj = V2D(e.v[1]);
      // cout << "edge " << e << endl << " dofs " << di << " " << dj << endl;
      // cout << " mat etr di-di" << endl; print_tm(cout, A(di, di)); cout << endl;
      // cout << " mat etr dj-dj" << endl; print_tm(cout, A(dj, dj)); cout << endl;
      // cout << " mat etr di-dj" << endl; print_tm(cout, A(di, dj)); cout << endl;
      // after BBDC, diri entries are compressed and mat has no entry (multidim BDDC doesnt work anyways)

      Mat<FCC::DIM, FCC::DIM, double> negOD = -A(di, dj);
      Vec<FCC::DIM> tang = vdata[e.v[1]].pos - vdata[e.v[0]].pos;
      tang /= L2Norm(tang);

      // cout << "scaled tang " << endl << tang << endl;
      
      // l2-norm of t\circ t is now 1
      // t\circ t :: negOD = trans(T) negOD T
      Vec<FCC::DIM> odTang = negOD * tang;
      double const fac = abs(InnerProduct(odTang, tang));

      // cout << " negOD " << endl; print_tm(cout, negOD); cout << endl;
      // cout << " odTang " << endl << odTang << endl;
      // cout << " fac = " << fac << endl << endl;

      auto & emat = edata[e.id]; emat = 0.0;
      Iterate<FCC::ENERGY::DISPPV>([&](auto i) LAMBDA_INLINE {
        Iterate<FCC::ENERGY::DISPPV>([&](auto j) LAMBDA_INLINE {
          emat(i.value, j.value) = fac * tang(i.value) * tang(j.value);
        });
      });
      
      // cout << " -> emat = " << endl; print_tm(cout, emat); cout << endl;

      // double const alpha = leastSquaresFit(emat, A(di, dj));

      // auto fsem = fabsum(emat);
      // // cout << " initial fabsum " << fsem << endl;
      // emat *= etrs / fsem;
      // scale rotations with 1/h - approx. same energy norm as displacements
      // const double linv = 1.0/len;
      // rot_scalings[e.v[0]] = (rot_scalings[e.v[0]] == 0.0) ? 1.0/len : min(rot_scalings[e.v[0]], linv);
      // rot_scalings[e.v[1]] = (rot_scalings[e.v[1]] == 0.0) ? 1.0/len : min(rot_scalings[e.v[1]], linv);
      rot_scalings[e.v[0]] = 1.0;
      rot_scalings[e.v[1]] = 1.0;
    }
  }
  else if (auto spm_tm = dynamic_pointer_cast<SparseMatrixTM<typename FCC::ENERGY::TM>>(spmat)) // disp+rot
  {
    const auto & ffds = *finest_freedofs;
    const auto & A(*spm_tm);
    for (auto & e : edges) {
      auto di = V2D(e.v[0]); auto dj = V2D(e.v[1]);
      // after BBDC, diri entries are compressed and mat has no entry (mult multidim BDDC doesnt work anyways)
      double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / FCC::ENERGY::DPV : 1e-4; // after BBDC, diri entries are compressed and mat has no entry 
      // double fc = (ffds.Test(di) && ffds.Test(dj)) ? fabsum(A(di, dj)) / sqrt(fabsum(A(di,di)) * fabsum(A(dj,dj))) / dofpv(C::DIM) : 1e-4;
      auto & emat = edata[e.id]; emat = 0;
      Iterate<FCC::ENERGY::DPV>([&](auto i) LAMBDA_INLINE { emat(i.value, i.value) = fc; });
    }
  }
  else
  {
    throw Exception("not sure how to compute edge weights from matrix!");
  }

  for (auto k : Range(rot_scalings))
    { rot_scalings[k] = (rot_scalings[k] == 0.0) ? 1.0 : rot_scalings[k]; }
  top_mesh->AllreduceNodalData<NT_VERTEX>(rot_scalings, [&](auto & in) { return std::move(min_table(in)); } );
  for (auto k : Range(vdata))
    { vdata[k].rot_scaling = rot_scalings[k]; }

  auto mesh = make_shared<typename FCC::TMESH>(std::move(*top_mesh), a, b);

  // std::ofstream os("finest_mesh.out");
  // os << *mesh << endl;

  return mesh;
} // VertexAMGPCBuildAlgMesh_ALG_scal


template<class FCC> template<class TD2V, class TV2D> shared_ptr<typename FCC::TMESH>
VertexAMGPC<FCC> :: BuildAlgMesh_ALG_blk (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat, TD2V D2V, TV2D V2D) const
{
  static Timer ti("BuildAlgMesh_ALG_blk"); RegionTimer rt(ti);
  const auto & O(static_cast<Options&>(*options));

  auto spm_tm = my_dynamic_pointer_cast<SparseMatrixTM<double>>(spmat, "BuildAlgMesh_ALG_blk - matrix");

  /** Vertex Data  **/
  auto a = new AttachedEVD<DIM>(Array<ElastVData<FCC::DIM>>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
  auto vdata = a->Data(); // TODO: get penalty dirichlet from row-sums (only taking x/y/z displacement entries)
  FlatArray<int> vsort = node_sort[NT_VERTEX];

  if (O.vertexSpecification == VertexAMGPCOptions::VERT_SPEC::FROM_MESH_NODES)
  {
    const auto & MA(*ma);
    Vec<FCC::DIM> t;
    for (auto k : Range(O.v_nodes)) {
      auto vnum = vsort[k];
      vdata[vnum].wt = 0;
      vdata[vnum].rot_scaling = 1.0;
      GetNodePos(O.v_nodes[k], MA, vdata[vnum].pos, t);
    }
  }
  else
  {
    throw Exception("Strict alg mode only for block-mat!");
  }

  /** Edge Data  **/
  auto b = new AttachedEED<FCC::DIM>(Array<ElasticityEdgeData<FCC::DIM>>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED); // !! has to be distr
  auto edata = b->Data();
  const auto& dof_blocks(O.block_s);
  auto edges = top_mesh->GetNodes<NT_EDGE>();
  const auto& ffds = *finest_freedofs;
  const auto & MAT = *spm_tm;

  Array<double> rot_scalings(top_mesh->GetNN<NT_VERTEX>()); rot_scalings = 0.0;

  for (const auto & e : edges) {
    auto dis = V2D(e.v[0]); auto djs = V2D(e.v[1]); auto diss = dis.Size();
    auto & ed = edata[e.id]; ed = 0;
    // after BBDC, diri entries are compressed and mat has no entry (mult multidim BDDC doesnt work anyways)
    if (ffds.Test(dis[0]) && ffds.Test(djs[0])) {
double x = 0, sumi = 0, sumj = 0;
// TODO: should I scale with diagonal inverse here ??
// actually, i think i should scale with diag inv, then sum up, then scale back
typename FCC::ENERGY::TM aij(0);
for (auto i : Range(dis)) // this could be more efficient
  { sumi += calc_trace(MAT(dis[i], dis[i])); }
for (auto j : Range(djs))
  { sumj += calc_trace(MAT(dis[j], dis[j])); }
for (auto i : Range(dis)) { // this could be more efficient
  x += fabs(MAT(dis[i], djs[i]));
  for (auto j = i+1; j < diss; j++)
    { x += 2*fabs(MAT(dis[i], djs[j])); }
}
x /= (diss * diss) * sqrt(sumi * sumj);
double len = 1.0;
if (diss == FCC::ENERGY::DISPPV) {
  Vec<FCC::DIM> tang = vdata[e.v[1]].pos - vdata[e.v[0]].pos;
  len = L2Norm(tang);
  tang /= len;
  Iterate<FCC::ENERGY::DISPPV>([&](auto i) LAMBDA_INLINE {
      Iterate<FCC::ENERGY::DISPPV>([&](auto j) LAMBDA_INLINE {
    ed(i.value, j.value) = x * tang(i.value) * tang(j.value);
  });
    });
}
else {
  for (auto j : Range(diss))
    { ed(j,j) = x; }
}
// scale rotations with 1/h - approx. same energy norm as displacements
const double linv = 1.0/len;
// rot_scalings[e.v[0]] = (rot_scalings[e.v[0]] == 0.0) ? 1.0/len : min(rot_scalings[e.v[0]], linv);
// rot_scalings[e.v[1]] = (rot_scalings[e.v[1]] == 0.0) ? 1.0/len : min(rot_scalings[e.v[1]], linv);
rot_scalings[e.v[0]] = 1.0; // (rot_scalings[e.v[0]] == 0.0) ? 1.0/len : min(rot_scalings[e.v[0]], linv);
rot_scalings[e.v[1]] = 1.0; // (rot_scalings[e.v[1]] == 0.0) ? 1.0/len : min(rot_scalings[e.v[1]], linv);
    }
    else { // does not matter what we give in here, just dont want nasty NaNs below, however this is admittedly a bit hacky
ed(0,0) = 0.00042;
    }
  }

  for (auto k : Range(rot_scalings))
    { rot_scalings[k] = (rot_scalings[k] == 0.0) ? 1.0 : rot_scalings[k]; }
  top_mesh->AllreduceNodalData<NT_VERTEX>(rot_scalings, [&](auto & in) { return std::move(min_table(in)); } );
  for (auto k : Range(vdata))
    { vdata[k].rot_scaling = rot_scalings[k]; }

  auto mesh = make_shared<typename FCC::TMESH>(std::move(*top_mesh), a, b);

  // std::ofstream os("finest_mesh.out");
  // os << *mesh << endl;

  return mesh;
} // VertexAMGPCBuildAlgMesh_ALG_blk


template<class FCC> shared_ptr<typename FCC::TMESH>
VertexAMGPC<FCC> :: BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh) const
{
  static Timer ti("BuildAlgMesh_TRIV"); RegionTimer rt(ti);
  const auto & O(static_cast<Options&>(*options));
  auto a = new AttachedEVD<FCC::DIM>(Array<ElastVData<FCC::DIM>>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); // !! otherwise pos is garbage
  auto vdata = a->Data(); // TODO: get penalty dirichlet from row-sums (only taking x/y/z displacement entries)
  FlatArray<int> vsort = node_sort[NT_VERTEX];
  Vec<FCC::DIM> t; const auto & MA(*ma);
  for (auto k : Range(O.v_nodes)) {
    auto vnum = vsort[k];
    vdata[vnum].wt = 0;
    vdata[vnum].rot_scaling = 1.0;
    GetNodePos(O.v_nodes[k], MA, vdata[vnum].pos, t);
  }
  auto b = new AttachedEED<FCC::DIM>(Array<ElasticityEdgeData<FCC::DIM>>(top_mesh->GetNN<NT_EDGE>()), CUMULATED);
  for (auto & x : b->Data()) { SetIdentity(x); }
  auto mesh = make_shared<typename FCC::TMESH>(std::move(*top_mesh), a, b);
  return mesh;
} // VertexAMGPC<FCC> :: BuildAlgMesh_TRIV


template<> template<>
shared_ptr<BaseDOFMapStep> INLINE VertexAMGPC<ElasticityAMGFactory<3>> :: BuildEmbedding_impl<2> (BaseAMGFactory::LevelCapsule const &cap)
{ return nullptr; }


template<class FCC>
template<int BSA>
shared_ptr<SparseMat<BSA, FCC::BS>>
VertexAMGPC<FCC>::
BuildED (size_t height, shared_ptr<TopologicMesh> mesh)
{
  static_assert( (BSA == 1) || (BSA == FCC::ENERGY::DISPPV) || (BSA == FCC::ENERGY::DPV),
      "BuildED with nonsensical N !");

  const auto & O(static_cast<Options&>(*options));

  typedef SparseMat<BSA, FCC::BS> TED;

  if (O.dof_ordering != Options::DOF_ORDERING::REGULAR_ORDERING)
    { throw Exception("BuildED only implemented for regular ordering"); }

  const auto & M(*mesh);

  if constexpr ( BSA == FCC::ENERGY::DPV ) // TODO: rot-flipping ?!
    { return nullptr; }
  else if constexpr ( BSA == FCC::ENERGY::DISPPV ) // disp -> disp,rot embedding
  {
    if ( O.with_rots || (O.block_s.Size() != 1) || (O.block_s[0] != 1) )
      { throw Exception("Elasticity BuildED: disp/disp+rot, block_s mismatch"); }

    Array<int> perow(M.template GetNN<NT_VERTEX>());
    perow = 1;

    auto E_D = make_shared<TED>(perow, M.template GetNN<NT_VERTEX>());

    for (auto k : Range(perow))
    {
      E_D->GetRowIndices(k)[0] = k;
      auto & v = E_D->GetRowValues(k)[0];
      v = 0; Iterate<FCC::ENERGY::DISPPV>([&](auto i) { v(i.value, i.value) = 1; });
    }
    return E_D;
  }
  else if ( BSA == 1 )
  {
    Array<int> perow(height); perow = 1;
    auto E_D = make_shared<TED>(perow, M.template GetNN<NT_VERTEX>());
    size_t row = 0, os_ri = 0;
    for (auto bs : O.block_s) {
      for (auto k : Range(M.template GetNN<NT_VERTEX>()))
      {
        for (auto j : Range(bs))
        {
          E_D->GetRowIndices(row)[0] = k;
          E_D->GetRowValues(row)[0] = 0;
          E_D->GetRowValues(row)[0](os_ri + j) = 1;
          row++;
        }
      }
      os_ri += bs;
    }
    return E_D;
  }

} // VertexAMGPC<FCC> :: BuildED

#ifdef FILE_AMG_ELAST_2D_CPP
template<> void VertexAMGPC<ElasticityAMGFactory<2>> :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const
{
  auto& A = static_cast<ElasticityAMGFactory<2>::TSPM_TM&>(*mat);
  if ( (pardofs != nullptr) && (pardofs->GetDistantProcs().Size() != 0) ) {
    Array<int> is_zero(A.Height());
    for(auto k : Range(A.Height()))
{ is_zero[k] = (fabs(A(k,k)(2,2)) < 1e-10 ) ?  1 : 0; }
    AllReduceDofData(is_zero, NG_MPI_SUM, pardofs);
    for(auto k : Range(A.Height()))
if ( (pardofs->IsMasterDof(k)) && (is_zero[k] != 0) )
  { A(k,k)(2,2) = 1; }
  }
  else {
    for(auto k : Range(A.Height())) {
auto & diag_etr = A(k,k);
if (fabs(diag_etr(2,2)) < 1e-8)
  { diag_etr(2,2) = 1; }
    }
  }
} // ElasticityAMGFactory<DIM>::RegularizeMatrix
#endif // FILE_AMG_ELAST_2D_CPP

#ifdef FILE_AMG_ELAST_3D_CPP
template<> void VertexAMGPC<ElasticityAMGFactory<3>> :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const
{
  auto& A = static_cast<ElasticityAMGFactory<3>::TSPM_TM&>(*mat);
  typedef ElasticityAMGFactory<3>::TM TM;
  if ( (pardofs != nullptr) && (pardofs->GetDistantProcs().Size() != 0) ) {
    Array<TM> diags(A.Height());
    for(auto k : Range(A.Height()))
{ diags[k] = A(k,k); }
    MyAllReduceDofData(*pardofs, diags, [](auto & a, const auto & b) LAMBDA_INLINE { a+= b; });
    for(auto k : Range(A.Height())) {
if (pardofs->IsMasterDof(k)) {
  auto & dg_etr = A(k,k);
  dg_etr = diags[k];
  // cout << " REG DIAG " << k << endl;
  // RegTM<3,3,6>(dg_etr); // might be buggy !?
  RegTM<0,6,6>(dg_etr);
  // cout << endl;
}
else
  { A(k,k) = 0; }
    }
  }
  else {
    for(auto k : Range(A.Height())) {
// cout << " REG DIAG " << k << endl;
// RegTM<3,3,6>(A(k,k));
RegTM<0,6,6>(A(k,k));
    }
  }
} // ElasticityAMGFactory<DIM>::RegularizeMatrix
#endif // FILE_AMG_ELAST_3D_CPP

} // namespace amg

#endif // FILE_ELAST_PC_IMPL_HPP
