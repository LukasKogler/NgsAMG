#define FILE_AMGH1_CPP

#include "amg.hpp"
#include "amg_precond_impl.hpp"

namespace amg
{

  H1AMG :: H1AMG (shared_ptr<H1AMG::TMESH> mesh,  shared_ptr<H1AMG::Options> opts)
    : VWiseAMG<H1AMG, H1AMG::TMESH, 1>(mesh, opts)
  {
    options->block_s = {1};
  }

  H1AMG :: H1AMG (shared_ptr<BilinearForm> blf, const Flags & aflags, const string aname)
    : VWiseAMG<H1AMG, H1AMG::TMESH, 1>(mesh, opts)
  {
    options->block_s = {1};
  }


  shared_ptr<BaseSmoother> H1AMG :: BuildSmoother  (INT<3> level, shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> par_dofs,
						    shared_ptr<BitArray> free_dofs)
  {
    shared_ptr<const TSPM> spmat = dynamic_pointer_cast<TSPM> (mat);
    auto sm = make_shared<HybridGSS<1>> (spmat, par_dofs, free_dofs);
    sm->SetSymmetric(options->smooth_symmetric);
    return sm;
  }

  /** See ngsolve/comp/h1amg.cpp **/
  template<> void
  EmbedVAMG<H1AMG> :: AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
					ElementId ei, LocalHeap & lh)
  {
    if (options->energy != "ELMAT") return;
    // vertex weights
    static Timer t("EmbedVAMG<H1AMG>::AddElementMatrix");
    static Timer t1("EmbedVAMG<H1AMG>::AddElementMatrix - inv");
    static Timer t3("EmbedVAMG<H1AMG>::AddElementMatrix - v-schur");
    static Timer t5("EmbedVAMG<H1AMG>::AddElementMatrix - e-schur");
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
  }

  template<> shared_ptr<EmbedVAMG<H1AMG>::TMESH>
  EmbedVAMG<H1AMG> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh)
  {
    auto a = new H1VData(Array<double>(top_mesh->GetNN<NT_VERTEX>()), DISTRIBUTED);
    auto b = new H1EData(Array<double>(top_mesh->GetNN<NT_EDGE>()), DISTRIBUTED);
    if (options->energy == "TRIV") {
      a->Data() = 0.0; b->Data() = 1.0;
    }
    else if (options->energy == "ELMAT") {
      FlatArray<int> vsort = node_sort[NT_VERTEX];
      Array<int> rvsort(vsort.Size());
      for (auto k : Range(vsort.Size()))
	rvsort[vsort[k]] = k;
      auto ad = a->Data();
      for (auto key_val : *ht_vertex) {
	// int vnr = vsort[key_val.get<0>()];
	ad[rvsort[get<0>(key_val)]] = get<1>(key_val);
      }
      // cout << "elmat vertex weights: " << endl; prow2(ad); cout << endl;
      auto bd = b->Data();
      auto edges = top_mesh->GetNodes<NT_EDGE>();
      for (auto & e : edges) {
	bd[e.id] = (*ht_edge)[INT<2,int>(rvsort[e.v[0]], rvsort[e.v[1]]).Sort()];
      }
      // cout << "elmat edge weights: " << endl; prow2(bd); cout << endl;
    }
    else { // "ALG"
      // provisional for now, will probably only work on actually nodal mats
      FlatArray<int> vsort = node_sort[NT_VERTEX];
      Array<int> rvsort(vsort.Size());
      for (auto k : Range(vsort.Size()))
	rvsort[vsort[k]] = k;
      // sum up rows -> rest is vertex weight
      auto NV = vsort.Size();
      shared_ptr<BaseMatrix> fseqmat = finest_mat;
      if (auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat))
	fseqmat = fpm->GetMatrix();
      auto fspm = dynamic_pointer_cast<SparseMatrix<double>>(fseqmat);
      auto ad = a->Data();
      for (auto k : Range(NV)) {
	auto rvs = fspm->GetRowValues(k);
	double sum = 0; for (auto v : rvs) sum += v;
	ad[rvsort[k]] = sum;
      }
      // off-diag entry -> is edge weight
      auto edges = top_mesh->GetNodes<NT_EDGE>();
      const auto & cspm(*fspm);
      auto bd = b->Data();
      for (auto & e : edges) {
	bd[e.id] = fabs(cspm(rvsort[e.v[0]], rvsort[e.v[1]]));
      }
    }
    auto mesh = make_shared<H1AMG::TMESH>(move(*top_mesh), a, b);
    return mesh;
  }

  
  template<> shared_ptr<BaseDOFMapStep> EmbedVAMG<H1AMG> :: BuildEmbedding ()
  {
    auto & vsort = node_sort[NT_VERTEX];
    shared_ptr<ParallelDofs> fpds = finest_mat->GetParallelDofs();
    auto emb_mat = BuildPermutationMatrix<double>(vsort);
    if (on_dofs != nullptr) { // embed this
      Array<int> perow(fpds->GetNDofLocal());
      for (auto k : Rang(fpds->GetNDofLocal()))
	perow[k] = on_dofs->Test(k) ? 1 : 0;
      auto mat = make_shared<SparseMatrix<double>>(perow);
      int cnt = 0;
      for (auto k : Rang(fpds->GetNDofLocal()))
	if (on_dofs->Test(k)) {
	  mat->GetRowIndices(k) = cnt++;
	  mat->GetRowValues(k) = 1;
	}
      emb_mat = MatMultAB(*mat, *emb_mat);
    }
    else if (vsort.Size() != fpds->GetNDofLocal())
      { throw Exception("When seem to not be working on the full space, but we do not know where!"); }

    if (fpds->GetNDofLocal() != emb_mat->Height())
      throw Exception(string("EMBED MAT H does not fit: ") + to_string(fpds->GetNDofLocal()) + string("!=") + to_string(emb_mat->Height()));
    if (vsort.Size() != emb_mat->Width())
      throw Exception(string("EMBED MAT W does not fit: ") + to_string(vsort.Size()) + string("!=") + to_string(emb_mat->Height()));

    auto pmap = make_shared<ProlMap<SparseMatrixTM<double>>>(emb_mat, fpds, nullptr);

    return pmap;
  }

  
  RegisterPreconditioner<EmbedVAMG<H1AMG>> registet_h1amg_scal("ngs_amg.h1_scal");
  
} // namespace amg

#include "amg_tcs.hpp"

