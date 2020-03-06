#ifdef AUX_AMG

#ifndef FILE_AMG_FACET_AUX_HPP
#define FILE_AMG_FACET_AUX_HPP

namespace amg
{

  /** Auxiliary Elements **/

  /** An H1 Auxiliary Space "Element", consisting of constants as basis functions. **/
  template<int DIM>
  class FacetH1FE
  {
  public:
    FacetH1FE () { ; }
    FacetH1FE (NodeId facet_id, MeshAccess & ma) { ; }
    static constexpr int ND () { return DIM; }
    INLINE void CalcMappedShape (const BaseMappedIntegrationPoint & mip, 
				 SliceMatrix<double> shapes) const
    {
      shapes = 0;
      for (auto k : Range(DIM))
	{ shapes(k,k) = 1; }
    }
  };

#ifdef ELASTICITY
  /** An Elasticity Auxiliary Sspace "Element", consisting of rigid body modes as basis funcitons. **/
  template<int DIM>
  class FacetRBModeFE
  {
  protected:
    static constexpr int NDS = (DIM == 3) ? 3 : 2;
    static constexpr int NRT = (DIM == 3) ? 3 : 1;
    Vec<DIM> mid;
  public:
    FacetRBModeFE (NodeId facet_id, MeshAccess & ma) {
      Vec<DIM> t;
      GetNodePos<DIM>(facet_id, ma, mid, t);
    }
    // FacetRBModeFE (shared_ptr<MeshAccess> ma, int facetnr)
    // {
    //   Array<int> pnums;
    //   ma->GetFacetPNums(facetnr, pnums);
    //   mid = 0;
    //   for (auto pnum : pnums)
    // 	{ mid += 1.0/pnums.Size() * ma->GetPoint<DIM>(pnum); }
    // }
    static constexpr int ND () { return (DIM == 3) ? 6 : 3; }
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
	Vec<DIM> ei, cross;
	for (auto k : Range(NRT)) {
	  ei = 0; ei(k) = 1;
	  cross = Cross(ei, x_m_c);
	  for (auto l : Range(DIM))
	    { shapes(NDS + k, l) = cross(l); }
	}
      }
    }
  }; // class FacetRBModeFE

  /** END Auxiliary Elements **/
#endif


  /** FacetAuxSystem **/

  template<int DIM, class ASPACEA, class ASPACEB, class AAUXFE>
  class FacetAuxSystem
  {
  public:
    using SPACEA = ASPACEA;
    using SPACEB = ASPACEB;
    using AUXFE = AAUXFE;

    static constexpr int DPV = AUXFE::ND();

    using TM = Mat<DPV, DPV, double>;
    using TV = Vec<DPV, double>;
    using TPMAT = SparseMatrix<Mat<1,DPV,double>>;
    using TPMAT_TM = SparseMatrixTM<Mat<1,DPV,double>>;
    using TAUX = SparseMatrix<TM>;
    using TAUX_TM = SparseMatrixTM<TM>;

  protected:
    /** Compound space stuff **/
    shared_ptr<BilinearForm> bfa;
    shared_ptr<MeshAccess> ma;
    shared_ptr<CompoundFESpace> comp_fes;
    shared_ptr<ParallelDofs> comp_pds;        /** without MPI these are dummy pardofs **/
    shared_ptr<BitArray> comp_fds;            /** freedofs for the compound space **/
    shared_ptr<BaseMatrix> comp_mat;          /** compound space matrix **/

    /** Where spacea and spaceb are located in the compound space **/
    size_t ind_sa, os_sa, ind_sb, os_sb;
    shared_ptr<SPACEA> spacea;
    shared_ptr<SPACEB> spaceb;

    /** Auxiliary space stuff **/
    shared_ptr<ParallelDofs> aux_pds;      /** ParallelDofs for auxiliary space **/
    shared_ptr<BitArray> aux_fds;              /** auxiliary space freedofs **/
    // shared_ptr<BitArray> aux_free_verts;    /** dirichlet-vertices in aux space **/
    shared_ptr<TPMAT> pmat;                    /** aux-to compound-embedding **/
    shared_ptr<TAUX> aux_mat;                  /** auxiliary space matrix **/

    /** book-keeping **/
    int apf = 0, ape = 0, bpf = 0, bpe = 0; /** A/B-DOFs per facet/edge **/
    bool has_e_ctrbs = false;
    bool has_a_e = false, has_b_e = false;
    Table<int> flo_a_f, flo_a_e;            /** SpaceA DofNrs for each facet/edge/full facet **/
    Table<int> flo_b_f, flo_b_e;            /** SpaceB DofNrs for each facet/edge/full facet **/
    Array<double> facet_mat_data;           /** Facet matrix buffer **/

    Array<int> a2f_facet, f2a_facet;        /** all facets <-> fine facets mappings **/
    shared_ptr<BitArray> fine_facet;       /** is facet a fine facet ? **/
    
    /** Facet matrices: [a_e, a_f, a_e, a_f]^T \times [aux_f] **/
    Array<FlatMatrix<double>> facet_mat;

    bool elmat_sc = false;  // hacky for now

  public:

    FacetAuxSystem (shared_ptr<BilinearForm> _bfa);

    shared_ptr<CompoundFESpace> GetCompSpace () const { return comp_fes; }
    shared_ptr<FESpace> GetSpaceA () const { return spacea; }
    int GetIndA () const { return ind_sa; }
    int GetOsA () const { return os_sa; }
    shared_ptr<FESpace> GetSpaceB () const { return spaceb; }
    int GetIndB () const { return ind_sb; }
    int GetOsB () const { return os_sb; }
    shared_ptr<BaseMatrix> GetCompMat () const { return comp_mat; }
    shared_ptr<ParallelDofs> GetCompParDofs () const { return comp_pds; }
    shared_ptr<BitArray> GetCompFreeDofs () const { return comp_fds; }
    shared_ptr<TPMAT> GetPMat () const { return pmat; }
    shared_ptr<TAUX> GetAuxMat () const { return aux_mat; }
    shared_ptr<ParallelDofs> GetAuxParDofs () const { return aux_pds; }
    shared_ptr<BitArray> GetAuxFreeDofs () const { return aux_fds; } // free_verts are after sorting
    Array<Array<shared_ptr<BaseVector>>> GetRBModes () const;
    shared_ptr<BitArray> GetFineFacets () const { return fine_facet; }
    FlatArray<int> GetFMapA2F () const { return a2f_facet; }
    FlatArray<int> GetFMapF2A () const { return f2a_facet; }

    virtual shared_ptr<BaseVector> CreateAuxVector () const;

    virtual void Initialize (shared_ptr<BitArray> freedofs);
    virtual void Finalize (shared_ptr<BaseMatrix> _comp_mat);

    virtual void AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				   ElementId ei, LocalHeap & lh);

  protected:

    void AllocAuxMat ();
    void SetUpFacetMats ();
    void SetUpAuxParDofs ();
    void BuildPMat ();

    /** Pick out low-order DOFs of node_id **/
    template<class TLAM> INLINE void ItLO_A (NodeId node_id, Array<int> & dnums, TLAM lam);
    template<class TLAM> INLINE void ItLO_B (NodeId node_id, Array<int> & dnums, TLAM lam);

    template<ELEMENT_TYPE ET> INLINE
    void CalcFacetMat (ElementId vol_elid, int facet_nr, FlatMatrix<double> fmat, LocalHeap & lh);

    INLINE void Add_Facet (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
			   ElementId ei, LocalHeap & lh);

    INLINE void Add_Vol (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
			 ElementId ei, LocalHeap & lh);

    /** aux_elmat = P * elmat * PT **/
    INLINE void Add_Vol_simple (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				ElementId ei, LocalHeap & lh);

    /** Calc ker(PPT), regularize it. **/
    INLINE void Add_Vol_rkP (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
			     ElementId ei, LocalHeap & lh);

    INLINE void Add_Vol_elP (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
			     ElementId ei, LocalHeap & lh);

  }; // class FacetAuxSystem


  /** END FacetAuxSystem **/


  /** FacetAuxVertexAMGPC **/

  /** An Auxiliary Space Elasticty AMG Preconditioner, obtained by facet-wise embedding of
      the rigid body-modes. **/
  template<int ADIM, class AAUX_SYS, class AAMG_CLASS>
  class FacetAuxVertexAMGPC : public AAMG_CLASS
  {
  public:

    static constexpr int DIM = ADIM;

    using AUX_SYS = AAUX_SYS;
    using AMG_CLASS = AAMG_CLASS;
    using TMESH = typename AMG_CLASS::TMESH;

    class Options;

    using TM = typename AUX_SYS::TM;
    using TV = typename AUX_SYS::TV;
    using TPMAT = typename AUX_SYS::TPMAT;
    using TPMAT_TM = typename AUX_SYS::TPMAT_TM;
    using TAUX = typename AUX_SYS::TAUX;
    using TAUX_TM = typename AUX_SYS::TAUX_TM;

  protected:

    using Preconditioner::ma;

    /** Inherided from AMG_CLASS **/
    using AMG_CLASS::bfa, AMG_CLASS::options, AMG_CLASS::finest_freedofs, AMG_CLASS::finest_mat,
      AMG_CLASS::factory, AMG_CLASS::free_verts;
    using AMG_CLASS::use_v2d_tab, AMG_CLASS::d2v_array, AMG_CLASS::v2d_array, AMG_CLASS::v2d_table, AMG_CLASS::node_sort;
    using AMG_CLASS::amg_mat;

    shared_ptr<AUX_SYS> aux_sys;               /** the auxiliary space system **/

    bool __hacky_test = true;                  /** hacky **/

    shared_ptr<EmbeddedAMGMatrix> emb_amg_mat; /** as a preconditioner for the compound BLF **/

  public:

    /** Constructors **/

    FacetAuxVertexAMGPC (const PDE & apde, const Flags & aflags, const string aname = "precond")
      : AMG_CLASS (apde, aflags, aname)
    { throw Exception("PDE-Constructor not implemented!"); }

    FacetAuxVertexAMGPC (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string name = "precond");

    /** New methods **/

    shared_ptr<AUX_SYS> GetAuxSys () const { return aux_sys; }
    shared_ptr<typename AUX_SYS::TPMAT> GetPMat () const { return aux_sys->GetPMat(); }
    shared_ptr<typename AUX_SYS::TAUX> GetAuxMat () const { return aux_sys->GetAuxMat(); }
    shared_ptr<BitArray> GetAuxFreeDofs () const { return aux_sys->GetAuxFreeDofs(); } // free_verts are after sorting
    Array<Array<shared_ptr<BaseVector>>> GetRBModes () const { return aux_sys->GetRBModes(); }

    virtual shared_ptr<BaseVector> CreateAuxVector () const { return aux_sys->CreateAuxVector(); }

    /** Inherited from Preconditioner/BaseMatrix **/
    virtual const BaseMatrix & GetAMatrix () const override;
    virtual const BaseMatrix & GetMatrix () const override;
    virtual shared_ptr<BaseMatrix> GetMatrixPtr () override;
    virtual void Mult (const BaseVector & b, BaseVector & x) const override;
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;
    virtual AutoVector CreateColVector () const override;
    virtual AutoVector CreateRowVector () const override;
    virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;
    virtual void FinalizeLevel (const BaseMatrix * mat) override;
    virtual void Update () override;

    /** Inherited from AMG_CLASS **/
    virtual void SetUpMaps () override;
    virtual void BuildAMGMat () override;
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (shared_ptr<TopologicMesh> mesh) override;
    // virtual shared_ptr<BaseSmoother> BuildSmoother (const BaseAMGFactory::AMGLevel & amg_level) override;
    virtual shared_ptr<BaseAMGPC::Options> NewOpts () override;
    virtual void SetDefaultOptions (BaseAMGPC::Options& O) override;
    virtual void ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix = "ngs_amg_") override;

    /** Additional stuff **/
    virtual shared_ptr<BaseSmoother> BuildFLS () const;
    virtual shared_ptr<BaseSmoother> BuildFLS2 () const;
    shared_ptr<EmbeddedAMGMatrix> GetEmbAMGMat () const;

    virtual void AddElementMatrix (FlatArray<int> dnums, const FlatMatrix<double> & elmat,
				   ElementId ei, LocalHeap & lh) override;

  }; // class FacetWiseAuxiliarySpaceAMG

  /** END FacetAuxVertexAMGPC **/


  /** Options **/

  class BaseFacetAMGOptions
  {
  public:
    /** General **/
    bool aux_only = false;            // Is this be a PC for the auxiliary space only

    /** Element matrix **/
    bool elmat_sc = false;            // Form schur-complements w.r.t aux-dofs on elmats

    /** Finest level Smoother **/
    // bool smooth_aux_l0 = false;    // TODO: If aux_only is false, also smooth finest level AUX space ??
    bool el_blocks = true;            // Use element-blocks (default is facet-blocks) [not great with MPI]
    bool f_blocks_sc = false;         // Use element-blocks with MPI facets removed by SC [GARBAGE!!]

  public:
    BaseFacetAMGOptions () { ; }    

    virtual void SetFromFlags (const Flags & flags, string prefix)
    {
      aux_only = flags.GetDefineFlagX("ngs_amg_aux_only").IsTrue();
      elmat_sc = !flags.GetDefineFlagX("ngs_amg_elmat_sc").IsFalse();
      el_blocks = !flags.GetDefineFlagX("ngs_amg_el_blocks").IsFalse();
      f_blocks_sc = flags.GetDefineFlagX("ngs_amg_fsc").IsTrue();
    } // BaseFacetAMGOptions :: SetFromFlags

  }; // class BaseFacetAMGOptions

  /** END Options **/


  /** Really ugly stuff **/

  /** Space -> Element **/
  template<class SPACE, ELEMENT_TYPE ET> struct STRUCT_SPACE_EL { typedef void fe_type; };


  template<class SPACE, ELEMENT_TYPE ET> using SPACE_EL = typename STRUCT_SPACE_EL<SPACE,ET>::fe_type;


  /** Does space have dual shapes? **/
  template<class SPACE> struct SPACE_DS_TRAIT : std::true_type
  {
    static constexpr bool take_tang   = false;
    static constexpr bool take_normal = false;
  };

  template<class TSPACE, class TELEM, class TMIP> INLINE void CSDS (const TELEM & fel, const TMIP & mip, FlatMatrix<double> s, FlatMatrix<double> sd)
  {
    fel.CalcMappedShape(mip, s);
    if constexpr (SPACE_DS_TRAIT<TSPACE>::value) {
	fel.CalcDualShape(mip, sd);
      }
    else {
      if constexpr(SPACE_DS_TRAIT<TSPACE>::take_tang) {
	const auto & nv = mip.GetNV();
	for (auto k : Range(s.Height()))
	  { s.Row(k) -= InnerProduct(s.Row(k), nv) * nv; }
      }
      else if constexpr (SPACE_DS_TRAIT<TSPACE>::take_normal) {
	const auto & nv = mip.GetNV();
	for (auto k : Range(s.Height()))
	  { s.Row(k) = InnerProduct(s.Row(k), nv) * nv; }
      }
      sd = s;
    }
  }

  constexpr NODE_TYPE FACET_NT (int DIM) {
    if (DIM == 2) { return NT_EDGE; }
    else { return NT_FACE; }
  }

  /** END Really ugly stuff **/

} // namespace amg

#endif

#endif // AUX_AMG
