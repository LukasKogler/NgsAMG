#include "amg.hpp"
#include <hcurlhofe.hpp> 
#include <hdivhofe.hpp> 

namespace amg
{

  template<int DIM>
  class FacetRBModeFE
  {
  protected:
    static constexpr int NDS = (DIM == 3) ? 3 : 2;
    static constexpr int NRT = (DIM == 3) ? 3 : 1;
    Vec<DIM> mid;
  public:
    FacetRBModeFE (Vec<3> amid) : mid(amid) { ; }
    FacetRBModeFE (shared_ptr<MeshAccess> ma, int facetnr)
    {
      Array<int> pnums;
      ma->GetFacetPNums(facetnr, pnums);
      mid = 0;
      for (auto pnum : pnums)
	{ mid += 1.0/pnums.Size() * ma->GetPoint<DIM>(pnum); }
    }
    INLINE void CalcMappedShape (const BaseMappedIntegrationPoint & mip, 
				 SliceMatrix<double> shapes) const
    {
      shapes = 0;
      for (auto k : Range(NDS))
	{ shapes(k,k) = 1; }
      Vec<DIM> x_m_c = mip.GetPoint() - mid;
      if constexpr (DIM == 2) {
	  shapes(2,0) = -x_m_c(1);
	  shapes(2,1) = x_m_c(0);
	}
      else {
	Vec<3> ei, cross;
	for (auto k : Range(NRT)) {
	  ei = 0; ei(k) = 1;
	  cross = Cross(ei, x_m_c);
	  for (auto l : Range(DIM))
	    { shapes(NDS + k, l) = cross(l); }
	}
      }
    }
  }; // class FacetRBModeFE


  template<int DIM>
  class TDNNS_AUX_AMG_Preconditioner : public Preconditioner
  {
  public:

    static constexpr int DPV () {
      if constexpr(DIM==3) { return 6; }
      else { return 3; }
    }

    using TM = Mat<DPV(), DPV(), double>;
    using TV = Vec<DPV(), double>;
    using TPMAT = SparseMatrix<Mat<1,DPV(),double>>;
    using TPMAT_TM = SparseMatrixTM<Mat<1,DPV(),double>>;
  protected:

    shared_ptr<BilinearForm> blf;

    int ind_hc, os_hc, ind_hd, os_hd;
    shared_ptr<CompoundFESpace> comp_fes;
    shared_ptr<HCurlHighOrderFESpace> hcurl;
    shared_ptr<HDivHighOrderFESpace> hdiv;

    shared_ptr<ParallelDofs> aux_pardofs;
    shared_ptr<BitArray> aux_freedofs;
    shared_ptr<TPMAT> pmat;
    shared_ptr<SparseMatrix<Mat<DPV(), DPV(), double>>> aux_mat;

    // shared_ptr<SparseMatrixTM<Mat<1,dofpv(DIM), double>>> amg_emb;

    /** facet matrices **/
    int hc_per_facetfacet;                  // (3d) how many HCurl-DOFs per edge in wirebasket (default: 1)
    int hd_per_facet, hc_per_facet;         // how many HDiv-DOFs per facet in wirebasket (default: 3 in 3d, 2 in 2d)
  public:
    Table<int> facet_lo_dofs;               // [k] = [ hd_facet_dofs, hc_edge_dofs, hc_facet_dofs ]
    Array<INT<3, int>> lors;                     // [n_hd, n_hc_e, n_hc_f]
  protected:
    Array<double> facet_mat_data;
  public:
    Array<FlatMatrix<double>> facet_mat;
  protected:
    Array<Vec<3>> facet_cos;

    // using FACTORY = ElasticityAMGFactory<D>;
  public:

    TDNNS_AUX_AMG_Preconditioner (const PDE & apde, const Flags & aflags, const string aname = "precond")
      : Preconditioner (&apde, aflags, aname)
    { throw Exception("PDE-Constructor not implemented!"); }

    TDNNS_AUX_AMG_Preconditioner (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name = "precond");

    virtual const BaseMatrix & GetAMatrix () const override
    { return blf->GetMatrix(); }
    virtual const BaseMatrix & GetMatrix () const override
    { return *this; }
    virtual shared_ptr<BaseMatrix> GetMatrixPtr () override
    { return blf->GetMatrixPtr(); }
    virtual void Mult (const BaseVector & b, BaseVector & x) const override
    { ; }
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override
    { ; }
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override
    { ; }
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override
    { ; }
    virtual AutoVector CreateColVector () const override;
    virtual AutoVector CreateRowVector () const override;
    virtual shared_ptr<BaseVector> CreateAuxVector () const;

    virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;

    virtual void FinalizeLevel (const BaseMatrix * mat) override;

    virtual void Update () override { ; };

    virtual void AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				   ElementId ei, LocalHeap & lh) override;

    virtual void AddElementMatrix1 (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				   ElementId ei, LocalHeap & lh);

    virtual void AddElementMatrix2 (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				    ElementId ei, LocalHeap & lh);

    virtual void AddElementMatrix3 (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				    ElementId ei, LocalHeap & lh);

    virtual void AddElementMatrix4 (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				    ElementId ei, LocalHeap & lh);

    shared_ptr<TPMAT> GetPMat () const { return pmat; }
    shared_ptr<SparseMatrix<Mat<DPV(),DPV(),double>>> GetAuxMat () const { return aux_mat; }
    shared_ptr<BitArray> GetAuxFreeDofs () const { return aux_freedofs; }
    Array<Array<shared_ptr<BaseVector>>> GetRBModes () const;

  protected:


    void BuildEmbedding ();
    void AllocAuxMat ();
    void SetUpAuxParDofs ();
    void SetUpFacetMats ();

    template<ELEMENT_TYPE ET> INLINE
    void CalcElP (ElementId vol_elid, FlatArray<int> udm, FlatMatrix<double> Pmat, LocalHeap & lh);

    template<ELEMENT_TYPE ET> INLINE
    void CalcFacetMat (ElementId vol_elid, int facet_nr, LocalHeap & lh,
		       FlatArray<int> hd_f_in_vol, FlatArray<int> hc_f_in_vol,
		       FlatArray<Array<int>> hd_e_in_vol, FlatArray<Array<int>> hd_e_in_f);

    template<ELEMENT_TYPE ET> INLINE
    void CalcFacetMat2 (ElementId vol_elid, int facet_nr, LocalHeap & lh,
			FlatArray<int> hd_f_in_vol, FlatArray<int> hc_f_in_vol,
			FlatArray<Array<int>> hd_e_in_vol, FlatArray<Array<int>> hd_e_in_f);

  }; // TDNNS_AUX_AMG_Preconditioner

} // namespace amg
