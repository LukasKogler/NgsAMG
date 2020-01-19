#ifndef FILE_AMGMAT
#define FILE_AMGMAT

#include "amg.hpp"
#include "amg_map.hpp"
#include "amg_smoother.hpp"

namespace amg
{

  class EmbeddedAMGMatrix;


  class AMGMatrix : public BaseMatrix
  {
    friend class EmbeddedAMGMatrix;
  protected:
    Array<shared_ptr<BaseVector>> res_level, x_level, rhs_level;
    shared_ptr<DOFMap> map;
    Array<shared_ptr<const BaseSmoother>> smoothers;
    // Array<shared_ptr<const BaseMatrix>> mats;
    bool drops_out = false;
    bool has_crs_inv = false;
    size_t n_levels = 0;
    size_t n_levels_glob = 0;
    shared_ptr<BaseMatrix> crs_inv;

  public:
    AMGMatrix (shared_ptr<DOFMap> _map, FlatArray<shared_ptr<BaseSmoother>> _smoothers);

    void SetCoarseInv (shared_ptr<BaseMatrix> _crs_inv);
    
    void SmoothV (BaseVector & x, const BaseVector & b) const;

    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
    virtual void Mult (const BaseVector & b, BaseVector & x) const override;
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;

    virtual int VHeight () const override { return smoothers[0]->Height(); }
    virtual int VWidth () const override { return smoothers[0]->Width(); }
    virtual AutoVector CreateVector () const override { return map->CreateVector(0); };
    virtual AutoVector CreateColVector () const override { return map->CreateVector(0); };
    virtual AutoVector CreateRowVector () const override { return map->CreateVector(0); };

    virtual bool IsComplex () const override { return false; }

    /** used for visualizing base functions **/
    void CINV (shared_ptr<BaseVector> x, shared_ptr<BaseVector> b) const;
    virtual size_t GetNLevels (int rank) const;
    virtual size_t GetNDof (size_t level, int rank) const;
    virtual void GetBF (size_t level, int rank, size_t dof, BaseVector & vec) const;
    // virtual void GetEV (size_t level, int rank, size_t k_num, BaseVector & vec) const;
  };
  

  class EmbeddedAMGMatrix : public BaseMatrix
  {
  protected:
    shared_ptr<BaseSmoother> fls;
    shared_ptr<AMGMatrix> clm;
    shared_ptr<BaseDOFMapStep> ds;
    shared_ptr<BaseVector> res;
    shared_ptr<BaseVector> xbuf;

  public:
    EmbeddedAMGMatrix (shared_ptr<BaseSmoother> _fls, shared_ptr<AMGMatrix> _clm, shared_ptr<BaseDOFMapStep> _ds)
      : fls(_fls), clm(_clm), ds(_ds)
    { res = ds->CreateVector(); xbuf = ds->CreateVector(); }

    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
    virtual void Mult (const BaseVector & b, BaseVector & x) const override;
    virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;

    virtual int VHeight () const override { return (fls == nullptr) ? ds->GetParDofs()->GetNDofLocal() : fls->Height(); }
    virtual int VWidth () const override  { return (fls == nullptr) ? ds->GetParDofs()->GetNDofLocal() : fls->Height(); }
    virtual AutoVector CreateVector () const override { return ds->CreateVector(); };
    virtual AutoVector CreateColVector () const override { return ds->CreateVector(); };
    virtual AutoVector CreateRowVector () const override { return ds->CreateVector(); };
    
    virtual bool IsComplex () const override { return false; }

    /** used for visualizing base functions **/
    void CINV (shared_ptr<BaseVector> x, shared_ptr<BaseVector> b) const;
    virtual size_t GetNLevels (int rank) const;
    virtual size_t GetNDof (size_t level, int rank) const;
    virtual void GetBF (size_t level, int rank, size_t dof, BaseVector & vec) const;
  };

} // namespace amg

#endif
