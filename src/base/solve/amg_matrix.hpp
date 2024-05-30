#ifndef FILE_AMGMAT
#define FILE_AMGMAT

#include <base.hpp>
#include <dof_map.hpp>
#include <base_smoother.hpp>

namespace amg
{

class EmbeddedAMGMatrix;


class AMGMatrix : public BaseMatrix
{
  friend class EmbeddedAMGMatrix;
  friend class AMGSmoother;
protected:
  Array<shared_ptr<BaseVector>> res_level, x_level, rhs_level;
  shared_ptr<DOFMap> map;
  Array<shared_ptr<BaseSmoother>> smoothers;
  // Array<shared_ptr<const BaseMatrix>> mats;
  bool drops_out = false;
  bool has_crs_inv = false;
  size_t n_levels = 0;
  size_t n_levels_glob = 0;
  shared_ptr<BaseMatrix> crs_inv, crs_mat;
  int vwb = 0;

public:
  AMGMatrix (shared_ptr<DOFMap> _map, FlatArray<shared_ptr<BaseSmoother>> _smoothers);

  virtual ~AMGMatrix () = default;

  void SetCoarseInv (shared_ptr<BaseMatrix> _crs_inv, shared_ptr<BaseMatrix> _crs_mat);

  INLINE void Smooth (BaseVector & x, const BaseVector & b) const {
    switch(vwb) {
    case(0) : { SmoothV(x, b); break; }
    case(1) : { SmoothW(x, b); break; }
    case(2) : { SmoothBS(x, b); break; }
    }
  }

  void SmoothV (BaseVector & x, const BaseVector & b) const;
  void SmoothW (BaseVector & x, const BaseVector & b) const;
  void SmoothBS (BaseVector & x, const BaseVector & b) const;

  void SmoothVFromLevel (int startlevel, BaseVector  &x, const BaseVector &b, BaseVector  &res,
        bool res_updated, bool update_res, bool x_zero) const;

  virtual void SetVWB (int _vwb) { vwb = _vwb; }
  virtual int GetVWB () const { return vwb; }

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
  virtual tuple<size_t, int> GetNDof (int level, int rank) const; // returns ND/BS
  virtual void GetBF (BaseVector & vec, int level, size_t dof, int comp, int rank, int onLevel=0) const;
  // virtual void GetEV (size_t level, int rank, size_t k_num, BaseVector & vec) const;

  shared_ptr<DOFMap> GetMap () const { return map; }
  FlatArray<shared_ptr<BaseSmoother>> GetSmoothers () const { return smoothers; }
  shared_ptr<const BaseSmoother> GetSmoother (int level) const { return ( level < smoothers.Size() ) ? smoothers[level] : nullptr; }
  shared_ptr<BaseSmoother>       GetSmoother (int level)       { return ( level < smoothers.Size() ) ? smoothers[level] : nullptr; }
  shared_ptr<const BaseMatrix>   GetMatrix   (int level) const { return ( level < smoothers.Size() ) ? smoothers[level]->GetAMatrix() : crs_mat; }
  shared_ptr<BaseMatrix>         GetMatrix   (int level)       { return ( level < smoothers.Size() ) ? smoothers[level]->GetAMatrix() : crs_mat; }
  shared_ptr<BaseMatrix> GetCINV () const { return crs_inv; }
  shared_ptr<BaseMatrix> GetCMat () const { return crs_mat; }

  /** returns [OC, OC_l0, OC_l1, ... ] **/
  Array<double> GetOC () const;

}; // class AMGMatrix


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

  ~EmbeddedAMGMatrix () = default;

  virtual void MultTrans (const BaseVector & b, BaseVector & x) const override;
  virtual void Mult (const BaseVector & b, BaseVector & x) const override;
  virtual void MultTransAdd (double s, const BaseVector & b, BaseVector & x) const override;
  virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override;

  virtual int VHeight () const override { return (fls == nullptr) ? ds->GetUDofs().GetND() : fls->Height(); }
  virtual int VWidth () const override  { return (fls == nullptr) ? ds->GetUDofs().GetND() : fls->Height(); }
  virtual AutoVector CreateVector () const override { return ds->CreateVector(); };
  virtual AutoVector CreateColVector () const override { return ds->CreateVector(); };
  virtual AutoVector CreateRowVector () const override { return ds->CreateVector(); };

  virtual bool IsComplex () const override { return false; }

  /** used for visualizing base functions **/
  void CINV (shared_ptr<BaseVector> x, shared_ptr<BaseVector> b) const;
  virtual size_t GetNLevels (int rank) const;
  virtual tuple<size_t, int> GetNDof (int level, int rank) const;
  virtual void GetBF (BaseVector & vec, int level, size_t dof, int comp, int rank) const;

  shared_ptr<BaseDOFMapStep> GetEmbedding () const { return ds; }
  shared_ptr<BaseSmoother> GetFLS () const { return fls; }
  shared_ptr<AMGMatrix> GetAMGMatrix () const { return clm; }
}; // class EmbeddedAMGMatrix


/** AMG as a smoother **/
class AMGSmoother : public BaseSmoother
{
protected:
  int start_level;
  bool singleFLS;
  shared_ptr<AMGMatrix> amg_mat;
public:

  AMGSmoother (shared_ptr<AMGMatrix> _amg_mat, int _start_level = 0, bool _singleFLS = false);

  ~AMGSmoother () = default;

  virtual void
  Smooth (BaseVector  &x, const BaseVector &b,
          BaseVector  &res, bool res_updated = false,
          bool update_res = true, bool x_zero = false) const override;


  virtual void
  SmoothBack (BaseVector  &x, const BaseVector &b,
              BaseVector &res, bool res_updated = false,
              bool update_res = true, bool x_zero = false) const override;
 

  virtual size_t GetNOps () const override;

}; // class AMGSmoother


} // namespace amg

#endif
