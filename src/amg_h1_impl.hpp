#ifndef FILE_AMG_H1_IMPL_HPP
#define FILE_AMG_H1_IMPL_HPP

#include "amg_h1.hpp"
#include "amg_energy_impl.hpp"
#include "amg_factory_nodal_impl.hpp"
#include "amg_factory_vertex_impl.hpp"
#include "amg_pc.hpp"
#include "amg_pc_vertex.hpp"
#include "amg_pc_vertex_impl.hpp"

namespace amg
{

  template<class TMAP> INLINE void H1VData :: map_data_impl (const TMAP & cmap, H1VData & ch1v) const
  {
    auto & cdata = ch1v.data;
    cdata.SetSize(cmap.template GetMappedNN<NT_VERTEX>()); cdata = 0.0;
    auto map = cmap.template GetMap<NT_VERTEX>();
    auto lam_v = [&](const AMG_Node<NT_VERTEX> & v)
      { auto cv = map[v]; if (cv != -1) cdata[cv] += data[v]; };
    bool master_only = (GetParallelStatus()==CUMULATED);
    mesh->Apply<NT_VERTEX>(lam_v, master_only);

    auto emap = cmap.template GetMap<NT_EDGE>();
    auto aed = get<1>(static_cast<H1Mesh*>(mesh)->Data()); aed->Cumulate(); // should be NOOP
    auto edata = aed->Data();
    mesh->Apply<NT_EDGE>( [&](const auto & e) LAMBDA_INLINE {
	if (emap[e.id] == -1) {
	  Iterate<2>([&](auto i) LAMBDA_INLINE {
	      if (map[e.v[i.value]] == -1) {
		auto cv = map[e.v[1-i.value]];
		if (cv != -1) {
		  cdata[cv] += edata[e.id];
		}
	      }
	    });
	}
      }, master_only);
    ch1v.SetParallelStatus(DISTRIBUTED);
  } // H1VData::map_data_impl


  template<class TMESH> INLINE void H1EData :: map_data_impl (const TMESH & cmap, H1EData & ch1e) const
  {
    auto & cdata = ch1e.data;
    cdata.SetSize(cmap.template GetMappedNN<NT_EDGE>()); cdata = 0.0;
    auto map = cmap.template GetMap<NT_EDGE>();
    auto lam_e = [&](const AMG_Node<NT_EDGE>& e)
      { auto cid = map[e.id]; if ( cid != decltype(cid)(-1)) { cdata[cid] += data[e.id]; } };
    bool master_only = (GetParallelStatus()==CUMULATED);
    mesh->Apply<NT_EDGE>(lam_e, master_only);
    ch1e.SetParallelStatus(DISTRIBUTED);
  } // H1EData :: map_data_impl
} // namespace amg


#ifdef FILE_AMGPC_HPP

namespace amg
{
  /** Need this only if we also include the PC headers **/

  template<class FACTORY, class HTVD, class HTED>
  void ElmatVAMG<FACTORY, HTVD, HTED> :: AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
							   ElementId ei, LocalHeap & lh)
  {
    const auto & O(static_cast<Options&>(*options));

    if (O.energy != Options::ENERGY::ELMAT_ENERGY)
      { return; }

    // vertex weights
    static Timer t("AddElementMatrix");
    static Timer t1("AddElementMatrix - inv");
    static Timer t3("AddElementMatrix - v-schur");
    static Timer t5("AddElementMatrix - e-schur");
    RegionTimer rt(t);
    size_t ndof = dnums.Size();
    BitArray used(ndof, lh);
    FlatMatrix<double> ext_elmat(ndof+1, ndof+1, lh);
    {
      ThreadRegionTimer reg (t5, TaskManager::GetThreadId());
      ext_elmat.Rows(0,ndof).Cols(0,ndof) = elmat;
      ext_elmat.Row(ndof) = 1;
      ext_elmat.Col(ndof) = 1;
      ext_elmat(ndof, ndof) = 0;
      CalcInverse (ext_elmat);
    }
    {
      RegionTimer reg (t1);
      for (size_t i = 0; i < dnums.Size(); i++)
        {
          Mat<2,2,double> ai;
          ai(0,0) = ext_elmat(i,i);
          ai(0,1) = ai(1,0) = ext_elmat(i, ndof);
          ai(1,1) = ext_elmat(ndof, ndof);
          ai = Inv(ai);
          double weight = fabs(ai(0,0));
          // vertex_weights_ht.Do(INT<1>(dnums[i]), [weight] (auto & v) { v += weight; });
          (*ht_vertex)[dnums[i]] += weight;
        }
    }
    {
      RegionTimer reg (t3);
      for (size_t i = 0; i < dnums.Size(); i++)
        for (size_t j = 0; j < i; j++)
          {
            Mat<3,3,double> ai;
            ai(0,0) = ext_elmat(i,i);
            ai(1,1) = ext_elmat(j,j);
            ai(0,1) = ai(1,0) = ext_elmat(i,j);
            ai(2,2) = ext_elmat(ndof,ndof);
            ai(0,2) = ai(2,0) = ext_elmat(i,ndof);
            ai(1,2) = ai(2,1) = ext_elmat(j,ndof);
            ai = Inv(ai);
            double weight = fabs(ai(0,0));
            // edge_weights_ht.Do(INT<2>(dnums[j], dnums[i]).Sort(), [weight] (auto & v) { v += weight; });
	    (*ht_edge)[INT<2, int>(dnums[j], dnums[i]).Sort()] += weight;
          }
    }
  } // EmbedWithElmats<H1AMGFactory, double, double>::AddElementMatrix

} // namespace amg

#endif


#ifdef FILE_AMGH1_CPP

namespace amg {
  /** Only need these int he h1-pc cpp files **/
  
  
  template<class FCC>
  void VertexAMGPC<FCC> :: SetDefaultOptions (BaseAMGPC::Options& base_O)
  {
    auto & O(static_cast<Options&>(base_O));

    auto ma = bfa->GetFESpace()->GetMeshAccess();

    /** Coarsening Algorithm **/
    O.crs_alg = Options::CRS_ALG::AGG;
    O.ecw_geom = false;
    O.n_levels_d2_agg = 1;
    O.agg_neib_boost = false;

    /** Smoothed Prolongation **/
    O.enable_sp = true;
    O.sp_needs_cmap = false;
    O.sp_min_frac = (ma->GetDimension() == 3) ? 0.08 : 0.15;
    O.sp_max_per_row = 3;
    O.sp_omega = 1.0;

    /** Discard **/
    O.enable_disc = true;
    O.disc_max_bs = 5;

    /** Level-control **/
    O.enable_multistep = true;
    O.use_static_crs = true;
    O.first_aaf = (ma->GetDimension() == 3) ? 0.05 : 0.1;
    O.aaf = 1/pow(2, ma->GetDimension());

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
    O.gs_ver = Options::GS_VER::VER3;

    /** FES-to-AMG Embedding **/
    O.block_s = { 1 }; // multi-dim / scalar
    // O.block_s = { 1, 1 (, 1) }; // compound
    // O.block_s = { 2/3 }; // reordered

  } // VertexAMGPC::SetDefaultOptions


  template<class FCC>
  void VertexAMGPC<FCC> :: ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix)
  {
    ;
  } // VertexAMGPC::ModifyOptions


  template<class FCC> template<class TD2V, class TV2D> shared_ptr<typename FCC::TMESH>
  VertexAMGPC<FCC> :: BuildAlgMesh_ALG_scal (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat,
					     TD2V D2V, TV2D V2D) const
  {
    static Timer t("BuildAlgMesh_ALG_scal"); RegionTimer rt(t);

    static_assert(is_same<int, decltype(D2V(0))>::value, "D2V mismatch");
    static_assert(is_same<int, decltype(V2D(0))>::value, "V2D mismatch");

    auto dspm = dynamic_pointer_cast<SparseMatrix<double>>(spmat);
    if (dspm == nullptr)
      { throw Exception("Could not cast sparse matrix!"); }

    // cout << "finest level mat: " << endl << *dspm << endl;

    const auto& cspm = *dspm;
    auto a = new H1VData(Array<double>(top_mesh->GetNN<NT_VERTEX>()), DISTRIBUTED); auto ad = a->Data(); ad = 0;
    auto b = new H1EData(Array<double>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED); auto bd = b->Data(); bd = 0;

    for (auto k : Range(top_mesh->GetNN<NT_VERTEX>()))
      { auto d = V2D(k); ad[k] = cspm(d,d); }

    auto edges = top_mesh->GetNodes<NT_EDGE>();
    auto& fvs = *free_verts;
    for (auto & e : edges) {
      auto di = V2D(e.v[0]); auto dj = V2D(e.v[1]);
      double v = cspm(di, dj);
      // cout << "edge " << e << " di dj " << di << " " << dj << ", val " << v << endl;
      // bd[e.id] = fabs(v) / sqrt(cspm(di,di) * cspm(dj,dj)); ad[e.v[0]] += v; ad[e.v[1]] += v;
      bd[e.id] = fabs(v); ad[e.v[0]] += v; ad[e.v[1]] += v;
    }
    
    for (auto k : Range(top_mesh->GetNN<NT_VERTEX>())) // -1e-16 can happen, is problematic
      { ad[k] = fabs(ad[k]); }

    auto mesh = make_shared<H1Mesh>(move(*top_mesh), a, b);

    // cout << " INIT MESH " << endl << endl << *mesh << endl << endl;
    // cout << " DiSTR VD " << endl; prow2(get<0>(mesh->Data())->Data()); cout << endl << endl;
    // cout << " DiSTR ED " << endl; prow2(get<1>(mesh->Data())->Data()); cout << endl << endl;
    // mesh->CumulateData();
    // cout << " CMUL VD " << endl; prow2(get<0>(mesh->Data())->Data()); cout << endl << endl;
    // cout << " CMUL ED " << endl; prow2(get<1>(mesh->Data())->Data()); cout << endl << endl;

    return mesh;
  } // VertexAMGPC::BuildAlgMesh_ALG_scal


  template<class FCC> template<class TD2V, class TV2D> shared_ptr<typename FCC::TMESH>
  VertexAMGPC<FCC> :: BuildAlgMesh_ALG_blk (shared_ptr<BlockTM> top_mesh, shared_ptr<BaseSparseMatrix> spmat, TD2V D2V, TV2D V2D) const
  {
    throw Exception("BuildAlgMesh_ALG_blk for H1 TODO (Comp/Reo-Comp spaces)!");
    return nullptr;
  } // VertexAMGPC::BuildAlgMesh_ALG_blk


  template<class FCC> shared_ptr<typename FCC::TMESH>
  VertexAMGPC<FCC> :: BuildAlgMesh_TRIV (shared_ptr<BlockTM> top_mesh) const
  {
    static Timer t("BuildAlgMesh_TRIV"); RegionTimer rt(t);
    /** vertex-weights are 0, edge-weihts are 1 **/
    auto a = new H1VData(Array<double>(top_mesh->GetNN<NT_VERTEX>()), CUMULATED); a->Data() = 0.0; 
    auto b = new H1EData(Array<double>(top_mesh->GetNN<NT_EDGE>()), CUMULATED); b->Data() = 1.0;
    auto mesh = make_shared<H1Mesh>(move(*top_mesh), a, b);
    return mesh;
  } // VertexAMGPC::BuildAlgMesh_TRIV


  template<class FCC> template<int BSA> shared_ptr<stripped_spm_tm<Mat<BSA, FCC::BS, double>>>
  VertexAMGPC<FCC> :: BuildED (size_t height, shared_ptr<TopologicMesh> mesh)
  {
    constexpr int BS = FCC::BS;
    if (BSA != BS)
      { throw Exception("This should not happen for H1!!"); }
    return nullptr;
  } // VertexAMGPC::BuildED

} // namespace amg

#endif

#endif
