#ifndef FILE_AMGPC_HPP
#define FILE_AMGPC_HPP

namespace amg
{
  /**
     This class handles the conversion from the original FESpace
     to the form that the Vertex-wise AMG-PC needs.

     Virtual class. 

     Needed for different numbering of DOFs or non-nodal base functions, 
     for example:
  	  - compound FESpaces [numbering by component, then node]
  	  - rotational DOFs [more DOFs, the "original" case]

     Implements:
       - Construct topological part of the "Mesh" (BlockTM)
       - Reordering of DOFs and embedding of a VAMG-style vector (0..NV-1, vertex-major blocks)
         into the FESpace.
     Implement in specialization:
       - Construct attached data for finest mesh
   **/
  template<class Factory>
  class EmbedVAMG : public Preconditioner
  {
  public:
    struct Options;

    EmbedVAMG (shared_ptr<BilinearForm> bfa, const Flags & aflags, const string aname = "precond");

    ~EmbedVAMG () { ; }

    virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;

    virtual void FinalizeLevel (const BaseMatrix * mat) override;

  protected:
    shared_ptr<BilinearForm> bfa;

    shared_ptr<BitArray> finest_freedofs, free_verts;
    shared_ptr<BaseMatrix> finest_matrix;

    shared_ptr<AMGMatrix> amg_mat;

    shared_ptr<Factory> factory;

    virtual void MakeOptionsFromFlags (const Flags & flags, string prefix = "ngs_amg_");

    virtual shared_ptr<BlockTM> BuildTopMesh (); // implemented once for all AMG_CLASS
    virtual shared_ptr<BlockTM> BTM_Mesh (shared_ptr<EQCHierarchy> eqc_h); // implemented once for all AMG_CLASS
    virtual shared_ptr<BlockTM> BTM_Alg (shared_ptr<EQCHierarchy> eqc_h); // implemented once for all AMG_CLASS
    virtual shared_ptr<BlockTM> BTM_Elmat (shared_ptr<EQCHierarchy> eqc_h)
    { throw Exception("Elmat topology not overloaded!"); };

    shared_ptr<Factory> BuildFactory (shared_ptr<TMESH> mesh);

    virtual shared_ptr<TMESH> BuildAlgMesh (shared_ptr<BlockTM> top_mesh); // implemented seperately for all AMG_CLASS
    virtual shared_ptr<TMESH> BuildInitialMesh () { return BuildAlgMesh(BuildTopMesh()); }
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding ();

    virtual void Finalize ();
    virtual void BuildAMGMat ();
  };

  /**
     basically just overloads addelementmatrix
   **/
  template<class Factory, class HTVD = double, class HTED = double>
  class EmbedWithElmats : public EmbedVAMG<Factory>
  {
    
  };

};

#endif
