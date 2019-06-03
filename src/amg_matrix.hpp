#ifndef FILE_AMGMAT
#define FILE_AMGMAT

namespace amg
{

  /**
     Used to treat level 0 seperately.
     Has transfer to coarse level + coarse level mat.
     Virtual method for what to do on level 0.
   **/
  // class TwoLevelMatrix : public BaseMatrix
  // {

  // };

  /**
     Used to treat all AMG levels.
     "Mult" does a V-cycle.
     
     Use TwoLevelMatrix + AMGMatrix for the coarse mat
     to treat level 0 differently.
  **/
  class AMGMatrix : public BaseMatrix
  {
  protected:
    Array<shared_ptr<BaseVector>> res_level, x_level, rhs_level;
    shared_ptr<DOFMap> map;
    Array<shared_ptr<const BaseSmoother>> smoothers;
    Array<shared_ptr<const BaseMatrix>> mats;
    bool drops_out = false;
    bool has_crs_inv = false;
    size_t n_levels = 0;
    size_t n_levels_glob = 0;
    shared_ptr<const BaseMatrix> crs_inv;
    Array<int> std_levels;
  public:
    AMGMatrix (shared_ptr<DOFMap> & _map,
	       Array<shared_ptr<const BaseSmoother>> & _smoothers,
	       Array<shared_ptr<const BaseMatrix>> & _mats)
      : map(_map), smoothers(_smoothers), mats(_mats)
    {
      n_levels = map->GetNLevels();
      n_levels_glob = map->GetParDofs(0)->GetCommunicator().AllReduce(n_levels, MPI_MAX);
      res_level.SetSize(map->GetNLevels());
      x_level.SetSize(map->GetNLevels());
      rhs_level.SetSize(map->GetNLevels());
      for(auto l:Range(map->GetNLevels())) {
	res_level[l] = map->CreateVector(l);
	// if (res_level[l]!=nullptr) res_level[l]->FVDouble() = 0.0;
	x_level[l] = map->CreateVector(l);
	// if (x_level[l]!=nullptr) res_level[l]->FVDouble() = 0.0;
	rhs_level[l] = map->CreateVector(l);
	// if (rhs_level[l]!=nullptr) res_level[l]->FVDouble() = 0.0;
      }
      if(mats.Last()==nullptr) drops_out = true;
      std_levels.SetSize(n_levels_glob+1);
      for(auto k:Range(n_levels_glob+1))
	std_levels[k] = k;
    }
    virtual AutoVector CreateVector () const override { return x_level[0]->CreateVector(); };
    void CINV(shared_ptr<BaseVector> x, shared_ptr<BaseVector> b);
    void SmoothV(const shared_ptr<BaseVector> & x,
  		 const shared_ptr<const BaseVector> & b) const
    { this->SmoothV(x, b, this->std_levels); };
    void SmoothV(const shared_ptr<BaseVector> & x,
  		 const shared_ptr<const BaseVector> & b,
		 FlatArray<int> on_levels) const;
    virtual void Mult (const BaseVector & b, BaseVector & x) const override {
      shared_ptr<BaseVector> sx(&x, NOOP_Deleter);
      shared_ptr<const BaseVector> sb(const_cast<BaseVector*>(&b), NOOP_Deleter);
      this->SmoothV(sx, sb);
    }
    virtual void MultAdd (double s, const BaseVector & b, BaseVector & x) const override {
      shared_ptr<const BaseVector> sb(const_cast<BaseVector*>(&b), NOOP_Deleter);
      this->SmoothV(x_level[0], sb);
      x += s * *x_level[0];
    }
    void AddFinalLevel (const shared_ptr<const BaseMatrix> & _crs_inv)
    { crs_inv = _crs_inv; has_crs_inv = true; }
    // void AddFinalLevel (const shared_ptr<const BaseSmoother> & crs_smoother_)
    void SetStdLevels(FlatArray<int> alevels)
    { std_levels.SetSize(alevels.Size()); std_levels = alevels; }
    virtual bool IsComplex() const override { return false; }
    virtual int VHeight() const override { return mats[0]->Height(); }
    virtual int VWidth() const override { return mats[0]->Width(); }
    virtual size_t GetNLevels(int rank) const {
      if(rank==0) return n_levels_glob;
      auto comm = map->GetParDofs(0)->GetCommunicator();
      return comm.AllReduce((comm.Rank()==rank) ? mats.Size() : 0, MPI_SUM);
    }
    // TODO: support this again...
    virtual size_t GetNDof(size_t level, int rank) const;
    virtual void GetBF(size_t level, int rank, size_t dof, BaseVector & vec) const;
    virtual void GetEV(size_t level, int rank, size_t k_num, BaseVector & vec) const;
  };
  
} // namespace amg

#endif
