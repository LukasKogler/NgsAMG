#include <base.hpp>
#include <python_ngstd.hpp>

// using namespace ngsolve;

#include "python_amg.hpp"

#include <amg_matrix.hpp>

namespace amg
{
    class SmootherBM2 : public BaseMatrix
    {
    protected:
    shared_ptr<BaseSmoother> sm;
    AutoVector res;
    bool sym;
    public:
    // SmootherBM(shared_ptr<BaseSmoother> _sm, bool _sym = true) : sm(_sm), sym(_sym) { res.AssignPointer(sm->CreateColVector()); }
    SmootherBM2(shared_ptr<BaseSmoother> _sm, bool _sym = true)
      : sm(_sm), res(std::move(_sm->CreateColVector())), sym(_sym)
    { ; }

    ~SmootherBM2() = default;

    virtual void Mult (const BaseVector & b, BaseVector & x) const override
    {
      BaseVector & r2 (const_cast<BaseVector&>(*res));
      x = 0.0;
      x.Cumulate();
      b.Distribute();
      r2 = b;
      // updated, update, zero
      if (sym) {
      sm->Smooth(x, b, r2, true, true, true);
      // sm->Smooth(x, b, r2, false, false, false);
      sm->SmoothBack(x, b, r2, false, false, false);
      }
      else
        { sm->Smooth(x, b, r2, false, false, false); }
    }
    virtual void MultTrans (const BaseVector & b, BaseVector & x) const override { Mult(b, x); }

    virtual int VHeight () const override { return sm->VHeight(); }
    virtual int VWidth () const override  { return sm->VWidth(); }
    // virtual AutoVector CreateVector () const override { return sm->CreateVector(); };
    virtual AutoVector CreateColVector () const override { return sm->CreateColVector(); };
    virtual AutoVector CreateRowVector () const override { return sm->CreateColVector(); };
    };


  void ExportSolve (py::module & m)
  {
    
    auto amgMatClass = py::class_<AMGMatrix, shared_ptr<AMGMatrix>, BaseMatrix>(m, "AMGMatrix", "AMG as a matrix");

    amgMatClass.def(py::init<>([](shared_ptr<DOFMap> dofMap,
                                          py::object pySmoothers,
                                          shared_ptr<BaseMatrix> cmat,
                                          shared_ptr<BaseMatrix> cinv) -> shared_ptr<AMGMatrix> {
      shared_ptr<AMGMatrix> amgMat = nullptr;
      
      auto smoothers = makeCArray<shared_ptr<BaseSmoother>>(pySmoothers);

      amgMat = make_shared<AMGMatrix>(dofMap, smoothers);
      if ( ( cmat != nullptr ) && ( cinv != nullptr ) )
      {
        amgMat->SetCoarseInv(cinv, cmat);
      }

      return amgMat;
    }),
    py::arg("map"),
    py::arg("smoothers") = py::list(),
    py::arg("cmat") = nullptr,
    py::arg("cinv") = nullptr);

    amgMatClass.def("GetMap", [&](AMGMatrix &amgMat) { return amgMat.GetMap(); });

    amgMatClass.def("GetNLevels", [&](AMGMatrix &amgMat) { return amgMat.GetNLevels(0); });

    amgMatClass.def("GetSmoother", [&](AMGMatrix &amgMat,
                                       int        level) -> shared_ptr<BaseSmoother> {
        return amgMat.GetSmoother(level);
    },
    py::arg("level") = 0
    );

    amgMatClass.def("GetMatrix", [&](AMGMatrix &amgMat,
                                       int        level) -> shared_ptr<BaseMatrix> {
        return amgMat.GetMatrix(level);
    },
    py::arg("level") = 0
    );

    amgMatClass.def("SubAMGMatrix", [&](shared_ptr<AMGMatrix> &amgMat,
                                        int level) -> shared_ptr<BaseMatrix> {
      shared_ptr<BaseMatrix> out = nullptr;
      if (level == 0)
        { out = amgMat; }
      else if ( level == amgMat->GetNLevels(0) )
        { out = amgMat->GetCINV(); }
      else {
        auto sm = make_shared<AMGSmoother>(amgMat, level);
        out = make_shared<SmootherBM2>(sm);
      }
      return out;
    },
    py::arg("start_level") = 0);

  }

} // namespace amg    