#include "amg.hpp"
#include "amg_tdnns.hpp"
#include <hdivhofe.hpp> 

namespace amg
{
  template<int DIM, class SPACEA, class SPACEB>
  class FacetWiseAuxiliarySpaceAMG : public Preconditioner
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
    
    /** For Auxiliary Space embedding **/
    Table<int> facet_lo_dofs;      // [k] = [ 1e, 1f, 2e, 2f ]
    Array<double> facet_mat_data;
    Array<FlatMatrix<double>> facet_mat;
    
    /** For AMG **/
    Array<Vec<3>> facet_cos;
    shared_ptr<ElasticityAMGFactory<DIM>> factory;

  public:

  protected:
  };

  template<int DIM>
  using TDNNS_AMG = FacetWiseAuxiliarySpaceAMG<DIM, HCurlHOFESpace, HDivHOFESpace>;

  template<int DIM>
  using MCS_AMG = FacetWiseAuxiliarySpaceAMG<DIM, HDivHOFESpace, VectorFacetFESpace>;

} // namespace amg

