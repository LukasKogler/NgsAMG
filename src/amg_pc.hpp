#ifndef FILE_AMGPC_HPP
#define FILE_AMGPC_HPP

namespace amg
{
  /**
     Base class AMG Preconditioner
   **/
  class BaseAMGPC : public Preconditioner
  {
  public:
    
    struct Options;

  protected:
    shared_ptr<Options> options;

    shared_ptr<BilinearForm> bfa;

    shared_ptr<BaseAMGFactory> factory;
    shared_ptr<BitArray> finest_freedofs;
    shared_ptr<BaseMatrix> finest_mat;
    shared_ptr<BaseMatrix> amg_mat;

  public:

    /** Constructors **/
    BaseAMGPC (const PDE & apde, const Flags & aflags, const string aname = "precond");
    BaseAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts = nullptr);

    ~BaseAMGPC ();

    /** Preconditioner overloads **/
    virtual void InitLevel (shared_ptr<BitArray> freedofs = nullptr) override;
    virtual void FinalizeLevel (const BaseMatrix * mat) override;

    /** BaseMatrix overloads **/
    virtual const BaseMatrix & GetAMatrix () const override;
    virtual const BaseMatrix & GetMatrix () const override;
    virtual shared_ptr<BaseMatrix> GetMatrixPtr () override;
    virtual void Mult (const BaseVector & b, BaseVector & x) const override;
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;

  protected:
    
    /** Options: construct, set default, set from flags, modify **/
    virtual shared_ptr<Options> MakeOptionsFromFlags (const Flags & flags, string prefix = "ngs_amg_");
    virtual shared_ptr<Options> NewOpts () = 0;
    virtual void SetOptionsFromFlags (Options& O, const Flags & flags, string prefix = "ngs_amg_");
    virtual void SetDefaultOptions (Options& O);
    virtual void ModifyOptions (Options & O);

    virtual shared_ptr<TopologicMesh> BuildInitialMesh () = 0;
    virtual shared_ptr<BaseAMGFactory> BuildFactory () = 0;
    
    virtual void BuildAMGMat ();

    virtual void InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level);
    virtual shared_ptr<BaseDOFMapStep> BuildEmbedding (shared_ptr<TopologicMesh> mesh) = 0;

  }; // BaseAMGPC

} // namespace amg

#endif
