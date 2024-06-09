#include "hdiv_hdg_embedding.hpp"

#include <hdivlofe.hpp> // dummy HDIV-FEs
#include <hdivhofe.hpp>
#include <fespace.hpp>
#include <compressedfespace.hpp>

namespace amg
{

template<int DIM>
auto
getTangentVectors(Vec<DIM, double> const &nv)
{
  if constexpr(DIM == 2)
  {
    Vec<DIM, double> t;
    t[0] = -nv[1];
    t[1] = nv[0];

    return t;
  }
  else
  {
    // e_l is unit vector such that nv[l] is the smallest entry
    int l = 0;

    if (abs(nv[1]) < abs(nv[0]))
    {
      l = 1;
    }

    if (abs(nv[2]) < abs(nv[l]))
    {
      l = 2;
    }

    Vec<DIM, double> el(0.0);
    el[l] = 1.0;

    // t0 = n \cross e_l
    Vec<DIM, double> t0 = Cross(el, nv);

    // t1 = n \cross t0
    Vec<DIM, double> t1 = Cross(t0, nv);

    return std::make_tuple(t0, t1);
  }
}

template<int ADIM, bool PRES_CONSTANTS>
class PreservedConstantsOnFacet
{
public:
  static constexpr int DIM  = ADIM;
  static constexpr int NDOF = PRES_CONSTANTS ? DIM : 0;

  PreservedConstantsOnFacet ()
  { ; }

  // first row is "n" and then come the constants
  INLINE
  void
  CalcMappedShape(BaseMappedIntegrationPoint const &mip,
                  SliceMatrix<double>               shapes) const
  {
    if constexpr(PRES_CONSTANTS)
    {
      Iterate<DIM>([&](auto k) {
        shapes.Row(k) = 0;
        shapes(k, k) = 1;
      });
    }
  }
}; // class PreservedConstantsOnFacet


template<int ADIM>
class PreservedDivFreeP1
{
public:
  static constexpr int DIM  = ADIM;
  static constexpr int NDOF = DIM * (DIM + 1) - 1; // 2d: 5, 3d: 11

  PreservedDivFreeP1 ()
  { ; }

  // first row is "n" and then come the constants
  INLINE
  void
  CalcMappedShape(BaseMappedIntegrationPoint const &mip,
                  SliceMatrix<double>               shapes) const
  {
    shapes = 0.0;

    // first DIM are the constants
    Iterate<DIM>([&](auto k) {
      shapes(k, k) = 1;
    });

    if constexpr(DIM == 3)
    {
      auto pt = mip.GetPoint();

      auto const x = pt[0];
      auto const y = pt[1];
      auto const z = pt[2];

      // (y, 0, 0), (z, 0, 0)
      shapes(3, 0) = y;
      shapes(4, 0) = z;

      // (0, x, 0), (0, z, 0)
      shapes(5, 1) = x;
      shapes(6, 1) = z;

      // (0, 0, x), (0, 0, y)
      shapes(7, 2) = x;
      shapes(8, 2) = y;

      // (x, -y, 0), (0, y, -z)
      shapes(9,  0) =  x;
      shapes(9,  1) = -y;
      shapes(10, 1) =  y;
      shapes(10, 2) = -z;
    }
    else
    {
      auto pt = mip.GetPoint();

      auto const x = pt[0];
      auto const y = pt[1];

      // (y, 0), (0, x)
      shapes(2, 0) = y;
      shapes(3, 1) = x;

      // (x, -y)
      shapes(4, 0) =  x;
      shapes(4, 1) = -y;
    }

  }
}; // class PreservedConstantsOnFacet


template<int ADIM, class APRESVECEL = PreservedConstantsOnFacet<ADIM, true>>
class NPlusConstantsOnFacet
{
public:
  using PRESVECEL = APRESVECEL;

  static constexpr int DIM        = ADIM;
  static constexpr int NDOF       = DIM;
  static constexpr int NUMPRES    = PRESVECEL::NDOF;
  static constexpr int HDIV_ORDER = 0;
  static constexpr int ND_TO_HDIV = 1;
  static constexpr int TF_ORDER   = 0;
  static constexpr int ND_TO_TF   = DIM - 1;

  bool const _flipOrientation;
  // double const _orientationFactor;

  NPlusConstantsOnFacet (MeshAccess const &MA,
                         int        const &facetNr,
                         bool       const &flipOrientation)
    : _flipOrientation(flipOrientation)
  { ; }

  // first row is "n" and then come the constants
  INLINE
  void
  CalcMappedShape(BaseMappedIntegrationPoint const &mip,
                  SliceMatrix<double>               shapes) const
  {
    // Vec<DIM, double> const &nv = static_cast<DimMappedIntegrationPoint<DIM> const &>(mip).GetNV();

    Vec<DIM, double> nv = static_cast<DimMappedIntegrationPoint<DIM> const &>(mip).GetNV();

    // consistent orientation of n AND tang-vecs
    if (_flipOrientation)
    {
      nv = -1.0 * nv;
    }

    nv /= L2Norm(nv);

    shapes = 0;
    shapes.Row(0) = nv;

    auto tang = getTangentVectors<DIM>(nv);

    if constexpr(DIM == 2)
    {
      shapes.Row(1) = tang;
    }
    else
    {
      shapes.Row(1) = get<0>(tang);
      shapes.Row(2) = get<1>(tang);
    }
  }

  INLINE
  Vec<NDOF, double>
  CalcFlow(double const &surf)
  {
    Vec<NDOF, double> flow = 0;
    // flow[0] = _flipOrientation ? -surf : surf;
    // the basis function is oriented the way the EDGE is oriented,
    // so always positive flow!
    flow[0] = surf;
    return flow;
  }

  // for k: lambda (k, loc-dof-nr)
  template<class TLAM> static INLINE void IterateLocalHDivDOFs(int kRow, TLAM lam)
  {
    // loc. DOF 0 is normal
    lam(0, 0);
  }

  template<class TLAM> static INLINE void IterateLocalTFDOFs(TLAM lam)
  {
    // loc. DOFs 1 (3d: and 2) are tangential
    lam(0, 1);

    if constexpr(DIM==3)
    {
      lam(1, 2);
    }
  }

  // // TODO: this could work for non-curved meshes
  // INLINE
  // void
  // CalcPresVecs(FlatMatrix<double> V)
  // {


  // }
};


/**
 * This one is really really iffy, probably super wrong, for curved facets...
 */
template<int ADIM, bool P1TF = false, class APRESVECEL = PreservedDivFreeP1<ADIM>>
class HDivHDGP1Facet
{
public:
  using PRESVECEL = APRESVECEL;

  static constexpr int DIM        = ADIM;
  static constexpr int HDIV_ORDER = 1;
  static constexpr int ND_TO_HDIV = DIM;
  static constexpr int TF_ORDER   = P1TF ? 1 : 0;
  static constexpr int ND_TO_TF   = (DIM - 1) * ( P1TF ? DIM : 1);
  static constexpr int NDOF       = ND_TO_HDIV + ND_TO_TF;
  static constexpr int NUMPRES    = PRESVECEL::NDOF;

  bool const _flipOrientation;
  // double const _orientationFactor;

  HDivHDGP1Facet (MeshAccess const &MA,
                  int        const &facetNr,
                  bool       const &flipOrientation)
    : _flipOrientation(flipOrientation)
  {
    Vec<DIM, double> tmp;
    GetNodePos<DIM>(NodeId(DIM == 2 ? NT_EDGE : NT_FACE, facetNr), MA, _midPoint, tmp);
  }

  // first row is "n" and then come the constants
  INLINE
  void
  CalcMappedShape(BaseMappedIntegrationPoint const &mip,
                  SliceMatrix<double>               shapes) const
  {
    shapes = 0;

    /**
     * Get tangent and normal vector such that they are consistent from both sides
     * of the facet
     **/
    Vec<DIM, double> nv = static_cast<DimMappedIntegrationPoint<DIM> const &>(mip).GetNV();

    if (_flipOrientation)
      { nv = -1.0 * nv; }

    nv /= L2Norm(nv);

    auto tang = getTangentVectors<DIM>(nv);

    // constant "n"
    shapes.Row(0) = nv;

    // linear, zero-flow "n" - probably "wrong" for curved facets
    Vec<DIM, double> distVec = mip.GetPoint() - _midPoint;

    if constexpr(DIM == 2)
    {
      // P1 normal

      // cout << " MP    "; prow(_midPoint); cout << endl;
      // cout << " point "; prow(mip.GetPoint()); cout << endl;
      // cout << " distV "; prow(distVec); cout << endl;
      // cout << " tang "; prow(tang); cout << endl;

      // maybe sth like this if we want it to scale better?
      // shapes.Row(1) = L2Norm(distVec)/mip.GetMeasure() * nv;
      double const dt0 = InnerProduct(distVec, tang); // norm(tang) = 1!
      shapes.Row(1) = dt0 * nv;

      // P0 tangential
      shapes.Row(ND_TO_HDIV) = tang;

      if constexpr(P1TF)
      {
        // P1 tangential
        shapes.Row(ND_TO_HDIV + 1) = dt0 * tang;
      }
    }
    else
    {
      // P1 normal
      double const dt0 = InnerProduct(distVec, get<0>(tang)); // norm(tang[0]) = 1!
      shapes.Row(1) = dt0 * nv;

      double const dt1 = InnerProduct(distVec, get<1>(tang)); // norm(tang[1]) = 1!
      shapes.Row(2) = dt1 * nv;

      // P0 tangential
      shapes.Row(ND_TO_HDIV    ) = get<0>(tang);
      shapes.Row(ND_TO_HDIV + 1) = get<1>(tang);

      if constexpr(P1TF)
      {
        // P1 tangential
        shapes.Row(ND_TO_HDIV + 2) = dt0 * get<0>(tang);
        shapes.Row(ND_TO_HDIV + 3) = dt0 * get<1>(tang);
        shapes.Row(ND_TO_HDIV + 4) = dt1 * get<0>(tang);
        shapes.Row(ND_TO_HDIV + 5) = dt1 * get<1>(tang);
      }
    }
  }

  INLINE
  Vec<NDOF, double>
  CalcFlow(double const &surf)
  {
    Vec<NDOF, double> flow = 0;
    // flow[0] = _flipOrientation ? -surf : surf;
    // the basis function is oriented the way the EDGE is oriented,
    // so always positive flow!
    flow[0] = surf;
    return flow;
  }

  // for k: lambda (k, loc-dof-nr)
  template<class TLAM> static INLINE void IterateLocalHDivDOFs(int kRow, TLAM lam)
  {
    if (kRow == 0) // row for p0 n-DOF
    {
      lam(0, 0);
    }
    else // row for p1 n-DOF
    {
      lam(0, 1);

      if constexpr(DIM == 3)
      {
        // no guarantee that the "tang" vecs I have here are the same as used for
        // construction of HDiv BFs, therefore both HDiv p1 depend on both aux-facet p1
        lam(1, 2);
      }
    }
  }

  template<class TLAM> static INLINE void IterateLocalTFDOFs(TLAM lam)
  {
    // tang p0 DOFs depend on p0 DOFs, but both can depend on both
    Iterate<DIM-1>([&](auto l) { lam(l.value, ND_TO_HDIV + l); });

    if constexpr(P1TF)
    {
      // tang p1 DOFs can depend on all p1 DOFs
      constexpr int OFF_P1 = DIM-1;
      Iterate<ND_TO_TF-OFF_P1>([&](auto l) { lam(OFF_P1 + l, ND_TO_HDIV + OFF_P1 + l); });
    }
  }

  // // TODO: this could work for non-curved meshes
  // INLINE
  // void
  // CalcPresVecs(FlatMatrix<double> V)
  // {
  // }

  protected:
    Vec<DIM, double> _midPoint;
};

template<int ADIM, class APRESVECEL = PreservedConstantsOnFacet<ADIM, false>>
class OnlyNOnFacet
{
public:
  static constexpr int DIM = ADIM;

  using PRESVECEL = APRESVECEL;

  static constexpr int NDOF       = 1;
  static constexpr int NUMPRES    = PRESVECEL::NDOF;
  static constexpr int HDIV_ORDER = 0;
  static constexpr int ND_TO_HDIV = 1;
  static constexpr int TF_ORDER   = 0;
  static constexpr int ND_TO_TF   = 0;

  bool const _flipOrientation;

  OnlyNOnFacet (MeshAccess const &MA,
                int        const &facetNr,
                bool       const &flipOrientation)
    : _flipOrientation(flipOrientation)
  { ; }

  // first row is "n" and then come the constants
  INLINE
  void
  CalcMappedShape(BaseMappedIntegrationPoint const &mip,
                  SliceMatrix<double>               shapes) const
  {
    Vec<DIM, double> nv = static_cast<DimMappedIntegrationPoint<DIM> const &>(mip).GetNV();

    // consistent orientation of n AND tang-vecs
    if (_flipOrientation)
    {
      nv = -1.0 * nv;
    }

    nv /= L2Norm(nv);

    shapes = 0;
    shapes.Row(0) = nv;
  }

  INLINE
  Vec<NDOF, double>
  CalcFlow(double const &surf)
  {
    Vec<NDOF, double> flow = 0;
    // flow[0] = _orientationFactor * surf;
    // the basis function is oriented the way the EDGE is oriented,
    // so always positive flow!
    flow[0] = surf;
    return flow;
  }

  template<class TLAM> static INLINE void IterateLocalHDivDOFs(int kRow, TLAM lam) { lam(0, 0); }
  template<class TLAM> static INLINE void IterateLocalTFDOFs(TLAM lam) { ; }
};


// pick out low-order DOFs from HDiv FACET-dofs, calls lambda on (locNum, dOFNum, dOFOrder)
template<int DIM, int ORDER, class TLAM>
INLINE
int
iterateHDivFacetDOFsWithOrder(NodeId nodeId, int p, FlatArray<int> dNums, TLAM lam)
{
  // unused, compressed, definedon, etc.
  if (dNums.Size() == 0)
    { return 0; }

  if constexpr(DIM == 3)
  {
    if (nodeId.GetType() == NT_EDGE)
      { return 0; }
  }

  /**
   *  HDiv low order face DOFs are, both 2d and 3d
   *    0              .. p0 DOF
   *    1              .. p1 DOF
   *  In 3d, we also have one of these, not sure which ine is correct
   *    1 + order      .. p1 DOF (3d)
   *    1 + p*(1+p)/2  .. p1 ODF (3d)
   */

  int cntLO = 0;

  auto callLam = [&](auto locNum, auto order) { lam(locNum, dNums[locNum], order); };

  // order 0 DOF
  callLam(0, 0);
  cntLO++;

  if constexpr(ORDER == 1)
  {
    callLam(1, 1);
    cntLO++;

    if constexpr(DIM == 3)
    {
      callLam(1+(p*(1+p))/2, 1);
      cntLO++;
    }
  }
  return cntLO;
}


template<int DIM, int ORDER, class TLAM>
INLINE
int
iterateHDivP1ElementDOFsWithOrder(int const facetPos, int const numFacets, TLAM lam)
{
  /**
   *  HDiv p1-element low order DOFs are
   *     - 1 (p0) per facet
   *     - 2 (p1) per facet    
   */

  // TODO: does this work for hex elements??
  //       Do quad facets have one extra "p1" DOF? I am not sure... 

  int cntLO = 0;

  // calls lambda w. (idx, numInDOFs, order)
  auto callLam = [&](auto locNum, auto order) { lam(cntLO++, locNum, order); };

  // order 0 DOF
  callLam(facetPos, 0);

  constexpr int P1_PER_FACET = DIM - 1;

  if constexpr(ORDER == 1)
  {
    Iterate<P1_PER_FACET>([&](auto l) {
      callLam(numFacets + P1_PER_FACET * facetPos + l, 1);
    });
  }

  return cntLO;
}

template<int DIM, int ORDER, class TLAM>
INLINE
int
iterateTangFacetDOFsWithOrder(NodeId nodeId, int p, FlatArray<int> dNums, TLAM lam)
{
  // unused, compressed, definedon, etc.
  if (dNums.Size() == 0)
    { return 0; }

  if constexpr(DIM == 3)
  {
    if (nodeId.GetType() == NT_EDGE)
      { return 0; }
  }

  int cntLO = 0;

  auto callLam = [&](auto idx, auto order) {
    lam(idx, dNums[idx], order);
    cntLO++;
  };

  /**
   *  3d TangentialFacetFESpace (in python: TangentialFacetFESpace) low order face DOFs are:
   *   0,1                  .. constant2
   *   1,2                  .. P1
   *   2*(p+1), 2*(p+1)+1   .. second p1
   */

  callLam(0, 0);

  if constexpr(DIM == 3)
  {
    callLam(1, 0);
  }

  if constexpr(ORDER == 1)
  {
    if constexpr(DIM == 3)
    {
      constexpr int P1_OFF_C0 = 2;
      
      // int const P1_OFF_C1 = P1_OFF_C0 + 2 * p;
      int const P1_OFF_C1 = 2*(1+p); // = 2 + 2*p

      callLam(P1_OFF_C0,     1);
      callLam(P1_OFF_C0 + 1, 1);
      callLam(P1_OFF_C1,     1);
      callLam(P1_OFF_C1 + 1, 1);
    }
    else
    {
      // I AM NOT SURE THIS IS CORRECT?
      callLam(1,   1);
    }
  }
  return cntLO;
}

/**
 * Converts:
 *   - AUX-SPACE -> NSPACES:  this gives (NFES X NAUX) output
 *   - NSPACES_REV -> AUX  :  this gives (NAUX x NFES) output
 * The latter is saved transposed so output can be put into a single FlatMatrix
 */
template<class AUX_FE, int NSPACES, int NSPACES_REVERSE, typename EVALFUNC, class TFMAT>
INLINE
std::tuple<double, Vec<AUX_FE::NDOF, double>>
calcFacetConvertMat (MeshAccess         const &MA,
                     int                const &facetNr,
                     bool               const &flipOrientation,
                     IVec<NSPACES + NSPACES_REVERSE, int>  const &fesND,
                     ElementId          const &volEID,
                     FESpace            const &fes,
                     EVALFUNC                  evalFESShapes,
                     TFMAT                     facetMat, // flat or slice
                     LocalHeap                &lh)
{
  HeapReset hr(lh);

  constexpr int DIM = AUX_FE::DIM;

  /** element/facet info, eltrans, geometry stuff **/
  const int ir_order = 3;
  ElementTransformation & eltrans = MA.GetTrafo (volEID, lh);
  auto el_facet_nrs = MA.GetElFacets(volEID);
  int locFacetNr = el_facet_nrs.Pos(facetNr);

  ELEMENT_TYPE et_vol = MA.GetElType(volEID);
  ELEMENT_TYPE et_facet = ElementTopology::GetFacetType (et_vol, locFacetNr);

  AUX_FE auxFE(MA, facetNr, flipOrientation);

  constexpr int auxND = AUX_FE::NDOF;

  const IntegrationRule & ir_facet = SelectIntegrationRule (et_facet, ir_order); // reference facet
  Facet2ElementTrafo facet_2_el(et_vol, BND); // reference facet -> reference vol
  IntegrationRule & ir_vol = facet_2_el(locFacetNr, ir_facet, lh); // reference VOL
  BaseMappedIntegrationRule & basemir = eltrans(ir_vol, lh);
  MappedIntegrationRule<DIM,DIM,double> & mir_vol(static_cast<MappedIntegrationRule<DIM,DIM,double>&>(basemir)); // mapped VOL

  /** buffers for shapes **/
  FlatMatrix<double> auxShape  (auxND, DIM, lh); // aux    BFs

  FlatMatrix<double> auxAux;

  if constexpr(NSPACES_REVERSE > 0)
  {
    auxAux.AssignMemory(auxND, auxND, lh);

    auxAux = 0.0;
  }

  IVec<NSPACES + NSPACES_REVERSE, FlatMatrix<double>> trialShape;//(fesND, DIM, lh); // primal BFs
  IVec<NSPACES, FlatMatrix<double>> testShape; //(fesND, DIM, lh); // dual   BFs

  IVec<NSPACES, FlatMatrix<double>> fesFes; // (fesND, fesND, lh);
  IVec<NSPACES, FlatMatrix<double>> fesAux; // (fesND, auxND,    lh);

  IVec<NSPACES_REVERSE, FlatMatrix<double>> auxFes; // (fesND, auxND,    lh);

  Iterate<NSPACES>([&](auto i)
  {
    trialShape[i].AssignMemory(fesND[i], DIM, lh);
    testShape[i].AssignMemory(fesND[i], DIM, lh);
    fesFes[i].AssignMemory(fesND[i], fesND[i], lh);
    fesAux[i].AssignMemory(fesND[i], auxND, lh);

    fesFes[i] = 0.0;
    fesAux[i] = 0.0;
  });

  Iterate<NSPACES_REVERSE>([&](auto i)
  {
    trialShape[NSPACES + i].AssignMemory(fesND[NSPACES + i], DIM, lh);
    auxFes[i].AssignMemory(auxND, fesND[NSPACES + i], lh);

    auxFes[i] = 0.0;
  });

  double facetSurf = 0.0;

  for (auto ip_nr : Range(mir_vol))
  {
    auto const &mip = mir_vol[ip_nr];

    evalFESShapes(mip, trialShape, testShape);

    auxFE.CalcMappedShape(mip, auxShape);

    if constexpr(NSPACES_REVERSE > 0)
    {
      auxAux += mip.GetWeight() * auxShape * Trans(auxShape);
    }

    // cout << " IP " << ip_nr << endl;
    // cout << " auxShape: " << endl << auxShape << endl;

    Iterate<NSPACES>([&](auto i)
    {

      // cout << " testShape[" << i << "] = " << endl << testShape[i] << endl;
      // cout << " trialShape[" << i << "] = " << endl << trialShape[i] << endl;

      /** fes-dual x fes **/
      fesFes[i] += mip.GetWeight() * testShape[i] * Trans(trialShape[i]);

      /** fes-dual x aux **/
      fesAux[i] += mip.GetWeight() * testShape[i] * Trans(auxShape);
    });

    Iterate<NSPACES_REVERSE>([&](auto i)
    {
      // cout << " trialShape[NSPACES + i] " << " = trialShape[" << NSPACES + i << "]: " << endl << trialShape[NSPACES + i] << endl;
      /** aux x fes, in aux space, trial==test! **/
      auxFes[i] += mip.GetWeight() * auxShape * Trans(trialShape[NSPACES + i]);
    });
    
    facetSurf += mip.GetWeight();
  }

  // cout << " SHAPES EVALUATED!" << std::endl;

  int offset = 0;

  Iterate<NSPACES>([&](auto i)
  {
    int const n = fesND[i];

    // cout << " fesFes[" << i << "]: " << endl << fesFes[i] << endl;
    // cout << " fesAux[" << i << "]: " << endl << fesAux[i] << endl;

    CalcInverse(fesFes[i]);

    // cout << " inv fesFes[" << i << "]: " << fesFes[i] << endl;

    facetMat.Rows(offset, offset + n) = fesFes[i] * fesAux[i];

    offset += n;
  });

  if constexpr(NSPACES_REVERSE > 0)
  {
    // cout << " auxAux: " << endl << auxAux << endl;

    CalcInverse(auxAux);

    // cout << " INV auxAux : " << endl << auxAux << endl;

    Iterate<NSPACES_REVERSE>([&](auto i)
    {
      int const n = fesND[NSPACES + i];

      // cout << " auxFes REV " << i << ": " << endl << auxFes[i] << endl;

      // save it transposed so we can use a single output-matrix
      facetMat.Rows(offset, offset + n) = Trans(auxFes[i]) * auxAux;

      offset += n;
    });
  }

  return std::make_tuple(facetSurf, auxFE.CalcFlow(facetSurf));
} // calcFacetConvertMat


/** HDivHDGEmbedding **/

HDivHDGEmbedding::
HDivHDGEmbedding(BaseStokesAMGPrecond const &stokesPre,
                 AUX_SPACE            const &auxSpace)
  : _stokesPre(stokesPre)
  , _auxSpace(auxSpace)
{
  FindSpaces();
} // HDivHDGEmbedding(..)


void
HDivHDGEmbedding::
FindSpaces()
{
  bool const useTFIfFound = (_auxSpace != RTZ);

  cout << "FindSpaces, _auxSpace = " << _auxSpace << endl;
  _fes = _stokesPre.GetBilinearForm()->GetFESpace();

  std::function<bool(int, IntRange, shared_ptr<FESpace>, shared_ptr<FESpace>)> checkSpace(
    [&](int idx, IntRange range, shared_ptr<FESpace> aSpace, shared_ptr<FESpace> setSpace) -> bool
      {
        if (auto hDivSpace = dynamic_pointer_cast<HDivHighOrderFESpace>(aSpace))
        {
          _hDivSpace = ( setSpace != nullptr ) ? setSpace : hDivSpace;
          _hDivIdx   = idx;
          _hDivRange = range;

          // cout << " _hDivSpace = " << _hDivSpace << endl;
          // cout << " _hDivIdx = " << _hDivIdx << endl;
          // cout << " _hDivRange = " << _hDivRange << endl;

          return true;
        }
        else if (auto tFSpace = dynamic_pointer_cast<TangentialFacetFESpace>(aSpace))
        {
          if (useTFIfFound)
          {
            _vFSpace = ( setSpace != nullptr ) ? setSpace : tFSpace;
            _vFIdx   = idx;
            _vFRange = range;
            // cout << " _vFSpace = " << _vFSpace << endl;
            // cout << " _vFIdx = " << _vFIdx << endl;
            // cout << " _vFRange = " << _vFRange << endl;
            return true;
          }
          else
          {
            return false;
          }
        }
        else if (auto cSpace = dynamic_pointer_cast<CompressedFESpace>(aSpace))
        {
          return checkSpace(idx, range, cSpace->GetBaseSpace(), ( setSpace != nullptr ) ? setSpace : cSpace);
        }
        return false;
      }
  );

  bool const singleHDivSpace = checkSpace(0, IntRange(0, _fes->GetNDof()), _fes, nullptr);

  if (singleHDivSpace)
  {
    _vFSpace = nullptr;
    _vFIdx   = -1;
    _vFRange = IntRange(0,0);

    std::cout << "HDivHDGEmbedding::FindSpaces, SINGLE HDIV!! " << std::endl;
  }
  else
  {
    int foundSpaces = 0;

    if (auto compFES = dynamic_pointer_cast<CompoundFESpace>(_fes))
    {
      for (auto k : Range(compFES->GetNSpaces()))
      {
        bool foundSpace = checkSpace(k,
                                     IntRange(compFES->GetRange(k)),
                                     compFES->Spaces()[k],
                                     nullptr);
        foundSpaces += foundSpace ? 1 : 0;
      }
    }

    std::cout << "HDivHDGEmbedding::FindSpaces, found " << foundSpaces << " spaces !! " << std::endl;
  }

  if ( ( GetTangFacetFESpace() == nullptr ) && (_auxSpace != RTZ) )
  {
    std::cout << " HDivHDGEmbedding without TF space, switching to RTZ aux-space (P0 without TF not implemented)!" << std::endl;
  }

  if ( (GetHDivFESpace().GetOrder() < 1) && (_auxSpace == P1) )
  {
    std::cout << " HDivHDGEmbedding on Order 0 HDiv space, switching from P1 to P0 aux-space!" << std::endl;
    _auxSpace = P0;
  }

} // HDivHDGEmbedding::FindSpaces


shared_ptr<MeshDOFs>
HDivHDGEmbedding::
CreateMeshDOFs(shared_ptr<BlockTM> fineMesh)
{
  auto const &FM = *fineMesh;

  auto const &MA = *GetHDivFESpace().GetMeshAccess();

  // int dofsPerFacet = 1 + ( GetTangFacetFESpace() != nullptr ? GetFESpace().GetMeshAccess()->GetDimension() : 0);
  int dofsPerFacet;

  int const DIM = GetFESpace().GetMeshAccess()->GetDimension();

  switch(_auxSpace)
  {
    case(RTZ): {
      if (DIM == 3)
      {
        dofsPerFacet = OnlyNOnFacet<3>::NDOF;
      }
      else
      {
        dofsPerFacet = OnlyNOnFacet<2>::NDOF;
      }
      break;
    }
    case(P0): {
      if (DIM == 3)
      {
        dofsPerFacet = NPlusConstantsOnFacet<3>::NDOF;
      }
      else
      {
        dofsPerFacet = NPlusConstantsOnFacet<2>::NDOF;
      }
      break;
    }
    case(P1): {
      auto tFOrder = GetTangFacetFESpace()->GetOrder();
      
      if (DIM == 3)
      {
        if (tFOrder == 0)
        {
          dofsPerFacet = HDivHDGP1Facet<3, false>::NDOF;
        }
        else
        {
          dofsPerFacet = HDivHDGP1Facet<3, true>::NDOF;
        }
      }
      else
      {
        if (tFOrder == 0)
        {
          dofsPerFacet = HDivHDGP1Facet<2, false>::NDOF;
        }
        else
        {
          dofsPerFacet = HDivHDGP1Facet<2, true>::NDOF;
        }
      }
      break;
    }
  }

  // int const dofsPerFacet = GetTangFacetFESpace() != nullptr ? GetFESpace().GetMeshAccess()->GetDimension() : 1;

  // ApplyToFE<int>(MA, facetNr, flipFacet, [&](auto const &auxFE) { return auxFE.GetND(); });

  Array<int> offsets(FM.template GetNN<NT_EDGE>() + 1);

  offsets = 0;

  // should have DOFs for exactly the edges corresponding to facets I have,
  // no DOFs on gg-edges!
  for (auto fnr : Range(MA.GetNFacets()))
  {
    auto const enr = getStokesPre().F2E(fnr);

    if (enr != -1)
    {
      offsets[1 + enr] = dofsPerFacet;
    }
  }

  auto meshDOFs = make_shared<MeshDOFs>(fineMesh);
  meshDOFs->SetOffsets(std::move(offsets));
  meshDOFs->Finalize(); // prefix sum happens in here!

  // cout << " CREATED MESHDOFS: " << endl;
  // cout << *meshDOFs << endl;

  return meshDOFs;
} // HDivHDGEmbedding::CreateMeshDOFs


std::tuple<shared_ptr<SparseMatrix<double>>,
            Array<double>, // facetSurf
            Array<double>, // facetFlow
            Array<shared_ptr<BaseVector>>> // vectors to preserve
HDivHDGEmbedding::
CreateDOFEmbedding(BlockTM  const &fMesh,
                   MeshDOFs const &meshDOFs)
{
  int const DIM = GetFESpace().GetMeshAccess()->GetDimension();

  if (DIM == 1)
  {
    throw Exception("1-DIM HDivHDGEmbedding makes no sense!");
    return std::make_tuple(nullptr, Array<double>(), Array<double>(), Array<shared_ptr<BaseVector>>());
  }
  
  switch(_auxSpace)
  {
    case(RTZ): {
      if (DIM == 3)
      {
        return CreateDOFEmbeddingImpl<OnlyNOnFacet<3>>(fMesh, meshDOFs);
      }
      else
      {
        return CreateDOFEmbeddingImpl<OnlyNOnFacet<2>>(fMesh, meshDOFs);
      }
      break;
    }
    case(P0): {
      if (DIM == 3)
      {
        return CreateDOFEmbeddingImpl<NPlusConstantsOnFacet<3>>(fMesh, meshDOFs);
      }
      else
      {
        return CreateDOFEmbeddingImpl<NPlusConstantsOnFacet<2>>(fMesh, meshDOFs);
      }
      break;
    }
    case(P1): {
      auto tFOrder = GetTangFacetFESpace()->GetOrder();
      
      if (DIM == 3)
      {
        if (tFOrder == 0)
        {
          return CreateDOFEmbeddingImpl<HDivHDGP1Facet<3, false>>(fMesh, meshDOFs);
        }
        else
        {
          return CreateDOFEmbeddingImpl<HDivHDGP1Facet<3, true>>(fMesh, meshDOFs);
        }
      }
      else
      {
        if (tFOrder == 0)
        {
          return CreateDOFEmbeddingImpl<HDivHDGP1Facet<2, false>>(fMesh, meshDOFs);
        }
        else
        {
          return CreateDOFEmbeddingImpl<HDivHDGP1Facet<2, true>>(fMesh, meshDOFs);
        }
      }
      break;
    }
    default: {
      throw Exception("HDivHDGEmbedding::CreateDOFEmbedding unsupported/unimplemented emb-type !?");
      return std::make_tuple(nullptr, Array<double>(), Array<double>(), Array<shared_ptr<BaseVector>>());
      break;
    }
  }
} // HDivHDGEmbedding::CreateDOFEmbedding

template<class FACETFE>
std::tuple<shared_ptr<SparseMatrix<double>>,
            Array<double>, // facetSurf
            Array<double>, // facetFlow
            Array<shared_ptr<BaseVector>>> // vectors to preserve
HDivHDGEmbedding::
CreateDOFEmbeddingImpl(BlockTM  const &fMesh,
                       MeshDOFs const &meshDOFs)
{
  LocalHeap lh(8388612, "JustAPen"); // 8MB + 4 bytes

  typedef typename FACETFE::PRESVECEL PRESEL;

  auto const &fes = *_fes;
  auto const &MA = *fes.GetMeshAccess();

  auto const & hDivFES = *_hDivSpace;

  bool const haveTangFacet = _vFSpace != nullptr;

  Array<int> allHDivDOFs;
  Array<int> hDivDOFs;
  Array<int> allLOHDivDOFs;
  Array<int> lODivDOFs;
  Array<int> allTangFacetDOFs;
  Array<int> tangFacetDOFs;
  Array<int> tangFacetShapePos(20); tangFacetShapePos.SetSize0();
  Array<int> embRowsHDiv(100); embRowsHDiv.SetSize0();
  Array<int> embRowsTF(100); embRowsTF.SetSize0();
  Array<int> facetEls(20); facetEls.SetSize0();

  Array<double> facetSurf(MA.GetNFacets());
  Array<double> facetFlow(MA.GetNFacets());

  constexpr int numPreserved = FACETFE::NUMPRES;

  Array<shared_ptr<BaseVector>> presVecs(numPreserved);

  for (auto k : Range(numPreserved))
  {
    presVecs[k] = make_shared<VVector<double>>(meshDOFs.GetNDOF());
  }

  // Array<int> hDivShapeRows(20);

  facetSurf = 0.0;
  facetFlow = 0.0;

  auto iterateFacetBlocks = [&](auto lam, bool const &computeP)
  {
    for (auto facetNr : Range(MA.GetNFacets()))
    {
      auto edgeNum = getStokesPre().F2E(facetNr);

      if (edgeNum == -1)
        { continue; }

      HeapReset hr(lh);

      NodeId facetId(NT_FACET, facetNr);

      // there MUST be some (local) vol el attached to the facet!
      MA.GetFacetElements(facetNr, facetEls);
      ElementId volEId(VOL, facetEls[0]);

      // std::cout << endl << "CONVERT FACET " << facetNr << ", els "; prow(facetEls); cout << " -> edge "
      //           << edgeNum << " = " << fMesh.template GetNode<NT_EDGE>(edgeNum) << std::endl;
      // std::cout << " computeP ? " << computeP << endl;

      // number of facet within the element's facets!
      auto elFacets = MA.GetElFacets(volEId);

      bool const flipOrientation = getStokesPre().EL2V(facetEls[0]) != fMesh.template GetNode<NT_EDGE>(edgeNum).v[0];

      embRowsHDiv.SetSize0();

      // bookkeeping HDiv component - we have custom order 1 (or 0) HDiv-FE
      auto &hDivFE   = hDivFES.GetFE(volEId, lh);

      // use the LO-element for computing shapes, that way we do not get into trouble
      // with stuff like unimplemented dual shapes (hodivfree)
      auto &lOHDivFE = GetHDivFE<FACETFE::DIM, FACETFE::HDIV_ORDER>(volEId, lh);

      hDivFES.GetDofNrs(volEId,  allHDivDOFs);
      hDivFES.GetDofNrs(facetId, hDivDOFs);

      // cout << " allHDivDOFs = "; prow2(allHDivDOFs); cout << endl;
      // cout << " hDivDOFs = "; prow2(hDivDOFs); cout << endl;

      int const nDOFHDiv  = hDivFE.GetNDof();

      unsigned hDivShapePos;

      int const nUsedHDiv = iterateHDivFacetDOFsWithOrder<FACETFE::DIM, FACETFE::HDIV_ORDER>
        (facetId,
         hDivFE.Order(),
         hDivDOFs,
         [&](auto j, auto dof, auto p)
         {
           embRowsHDiv.Append(_hDivRange.First() + dof);
         });

      hDivShapePos = elFacets.Pos(facetNr); // = locFacetNr, I think this is correct?

      // cout << " nDOFHDiv = " << nDOFHDiv << endl;
      // cout << " nUsedHDiv = " << nUsedHDiv << endl;
      // cout << " hDivShapePos = " << hDivShapePos << endl;
      // cout << " embRowsHDiv after HDIV: "; prow(embRowsHDiv); cout << endl;

      // bookkeeping tang-facet component
      int nUsedTangFacet = 0;


      if (haveTangFacet)
      {
        auto const & tangFacetFES = *_vFSpace;

        tangFacetFES.GetDofNrs(volEId,  allTangFacetDOFs);
        tangFacetFES.GetDofNrs(facetId, tangFacetDOFs);

        // cout << " allTangFacetDOFs = "; prow(allTangFacetDOFs); cout << endl;
        // cout << " tangFacetDOFs = "; prow(tangFacetDOFs); cout << endl;

        tangFacetShapePos.SetSize0();

        embRowsTF.SetSize0();

        // cout << " iterateTangFacetDOFsWithOrder " << endl;
        nUsedTangFacet = iterateTangFacetDOFsWithOrder<FACETFE::DIM, FACETFE::TF_ORDER>
                          (facetId,
                           tangFacetFES.GetOrder(),
                           tangFacetDOFs, [&](auto j, auto dof, auto p)
                           {
                            // cout << j << " " << dof << " " << p << ", offset " << _vFRange.First() << ", pos " << allTangFacetDOFs.Pos(dof) << endl;
                            // if (allTangFacetDOFs.Pos(dof) < 0)
                            // {
                            //   cout << "   allTangFacetDOFs "; prow(allTangFacetDOFs); cout << endl;
                            //   cout << "   DOF " << dof << endl;
                            // }
                             embRowsTF.Append(_vFRange.First() + dof);
                             tangFacetShapePos.Append(allTangFacetDOFs.Pos(dof));
                           });

        // cout << " nUsedTangFacet = " << nUsedTangFacet << endl;

        // cout << "   tangFacetShapePos= "; prow(tangFacetShapePos); cout << endl;

        // cout << " embRowsTF after TG: "; prow(embRowsTF); cout << endl;
      }

      int const nUsedFES = nUsedHDiv + nUsedTangFacet;
      int const nRowsP   = nUsedFES + numPreserved;

      // cout << " nUsedFES = " << nUsedFES << endl;
      // cout << " nRowsP = " << nRowsP << endl;

      FlatMatrix<double> P;

      double surf;
      Vec<FACETFE::NDOF, double> flow;

      if (computeP)
      {
        // cout << " nDOFHDiv = " << nDOFHDiv << endl;
        // cout << " lOHDivFE.GetNDof() = " << lOHDivFE.GetNDof() << endl;

        P.AssignMemory(nRowsP, FACETFE::NDOF, lh);

        // FlatMatrix<double> fullHDivShape(nDOFHDiv, FACETFE::DIM, lh);
        FlatMatrix<double> fullHDivShape(lOHDivFE.GetNDof(), FACETFE::DIM, lh);

        auto evalHDivShape = [&](auto const &mip,
                                 auto       &trialShape,
                                 auto       &testShape)
        {
          lOHDivFE.CalcMappedShape(mip, fullHDivShape);

          // cout << " iterateHDivP1ElementDOFsWithOrder " << endl;
          iterateHDivP1ElementDOFsWithOrder<FACETFE::DIM, FACETFE::HDIV_ORDER>
            (hDivShapePos, elFacets.Size(), [&](auto k, auto locNum, auto order)
            {
              // cout << k << " " << locNum << " " << order << endl;
              trialShape.Row(k) = fullHDivShape.Row(locNum);
            });

          // // !! this implicitly uses information abour ordering of DOFs in the
          // //    low-order HDIV element !!
          // trialShape.Row(0) = fullHDivShape.Row(hDivShapePos);
          // if constexpr(FACETFE::HDIV_ORDER == 1)
          // {
          //   // 1 per facet, then HO-dofs per facet
          //   trialShape.Row(1) = fullHDivShape.Row(elFacets.Size() + hDivShapePos);
          // }

          // cout << " fullHDivTrialShape = " << endl << fullHDivShape << endl;

          // lOHDivFE.CalcDualShape(mip, fullHDivShape);
          // testShape.Row(0)  = fullHDivShape.Row(hDivShapePos);

          // dual shapes not implemented for quads -> use normal trace
          const auto & nv = mip.GetNV();
          for (auto j : Range(trialShape.Height()))
          {
            testShape.Row(j) = InnerProduct(trialShape.Row(j), nv) * nv;
          }

          // cout << " HDIV trialShape: " << endl << trialShape << endl;
          // cout << " HDIV testShape: " << endl << testShape << endl;
        };

        PRESEL presEl;

        if (haveTangFacet)
        {
          auto const & tangFacetFES = *_vFSpace;

          auto &tangFacetFE = tangFacetFES.GetFE(volEId, lh);
          auto tangCF = tangFacetFES.GetEvaluator(VOL);

          tangFacetFES.GetDofNrs(facetId, tangFacetDOFs);

          int const nDOFTangFacet  = tangFacetFE.GetNDof();
          int const tangFacetOrder = tangFacetFES.GetOrder();

          FlatMatrix<double> fullTFacetShape(nDOFTangFacet, FACETFE::DIM, lh);

          auto [sbSurf, sbFlow] =
            calcFacetConvertMat<FACETFE, 2, 1>(
              MA,
              facetNr,
              flipOrientation,
              IVec<3, int>({ nUsedHDiv, nUsedTangFacet, PRESEL::NDOF }),
              volEId,
              fes,
              [&](auto const &mip,
                  IVec<3, FlatMatrix<double>> &trialShape,
                  IVec<2, FlatMatrix<double>> &testShape)
                {
                  evalHDivShape(mip,
                                trialShape[0],
                                testShape[0]);

                  tangCF->CalcMatrix(tangFacetFE, mip, Trans(fullTFacetShape), lh);

                  const auto & nv = mip.GetNV();

                  for (auto k : Range(fullTFacetShape.Height()))
                    { fullTFacetShape.Row(k) -= InnerProduct(fullTFacetShape.Row(k), nv) * nv; }

                  for (auto j: Range(tangFacetShapePos))
                  {
                    auto const pos = tangFacetShapePos[j];

                    trialShape[1].Row(j) = fullTFacetShape.Row(pos);
                    testShape[1].Row(j)  = fullTFacetShape.Row(pos);
                  }

                  // cout << " presEl.CalcMappedShape, trialShape[2] dims = " << endl;
                  // cout << trialShape[2].Height() << " x " << trialShape[2].Width() << endl;
                  presEl.CalcMappedShape(mip, trialShape[2]);
                  // cout << trialShape[2] << endl << endl;
                },
              P,
              lh);

          surf = sbSurf;
          flow = sbFlow;
        }
        else
        {
          auto [sbSurf, sbFlow] =
            calcFacetConvertMat<FACETFE, 1, 1>(
              MA,
              facetNr,
              flipOrientation,
              IVec<2, int>({ nUsedHDiv, PRESEL::NDOF }),
              volEId,
              fes,
              [&](auto const &mip,
                  IVec<2, FlatMatrix<double>> trialShape,
                  IVec<1, FlatMatrix<double>> testShape)
                  {
                    evalHDivShape(mip,
                                  trialShape[0],
                                  testShape[0]);

                    presEl.CalcMappedShape(mip, trialShape[1]);
                  },
              P,
              lh);

          surf = sbSurf;
          flow = sbFlow;
        }

        facetSurf[facetNr] = surf;
        facetFlow[facetNr] = flow[0];
      }
      else
      {
        P.AssignMemory(0, 0, lh);
      }

      // cout << " P: " << endl << P << endl;

      lam(edgeNum, embRowsHDiv, embRowsTF, P);

      // if (nUsedFES != FACETFE::NDOF)
      // {
      //   throw Exception("#DOFs aux-FE and used FES-FE per facet should match!");
      // }
    }
  };

  Array<int> perow(GetFESpace().GetNDof());
  perow = 0;

  // cout << " GetFESpace().GetNDof() = " << GetFESpace().GetNDof() << endl;

  iterateFacetBlocks([&](int enr,
                         FlatArray<int> embRowsHDiv,
                         FlatArray<int> embRowsTF,
                         FlatMatrix<double> facetMat)
  {
    for (auto kRow : Range(embRowsHDiv))
    {
      int cnt = 0;
      FACETFE::IterateLocalHDivDOFs(kRow, [&](auto l, auto col) { cnt++; });

      auto const row = embRowsHDiv[kRow];

      // perow[row] = FACETFE::ND_TO_HDIV;
      perow[row] = cnt;
    }

    for (auto kRow : Range(embRowsTF))
    {
      auto const row = embRowsTF[kRow];

      int cnt = 0;
      FACETFE::IterateLocalTFDOFs([&](auto l, auto col) { cnt++; });

      // perow[row] = FACETFE::ND_TO_TF;
      perow[row] = cnt;
    }
  }, false); // no need to compute facetMat

  auto embProl = make_shared<SparseMatrix<double>>(perow, meshDOFs.GetNDOF());

  perow = 0;

  iterateFacetBlocks([&](int enr,
                         FlatArray<int> embRowsHDiv,
                         FlatArray<int> embRowsTF,
                         FlatMatrix<double> facetMat)
  {
    // cout << " facetMat HDiv rows:" << endl;
    // cout << facetMat.Rows(0, embRowsHDiv.Size()) << endl;
    // cout << endl;

    auto const edgeDofs = meshDOFs.GetDOFNrs(enr);

    Iterate<FACETFE::ND_TO_HDIV>([&](auto kRow)
    {
      auto const row = embRowsHDiv[kRow];

      auto ris = embProl->GetRowIndices(row);
      auto rvs = embProl->GetRowValues(row);

      FACETFE::IterateLocalHDivDOFs(kRow, [&](auto l, auto col) {
        // cout << " row " << kRow << " -> " << row << ", l " << l << " -> col " << col << endl;
        ris[l] = meshDOFs.EdgeToDOF(enr, col);
        rvs[l] = facetMat(kRow, col);
      });
    });

    // cout << " facetMat TF rows:" << endl;
    // cout << facetMat.Rows(embRowsHDiv.Size(), embRowsHDiv.Size() + embRowsTF.Size()) << endl;
    // cout << endl;

    int tFOff = embRowsHDiv.Size();

    Iterate<FACETFE::ND_TO_TF>([&](auto kRow)
    {
      auto const row = embRowsTF[kRow];

      auto ris = embProl->GetRowIndices(row);
      auto rvs = embProl->GetRowValues(row);

      FACETFE::IterateLocalTFDOFs([&](auto l, auto col)
      {
        ris[l] = meshDOFs.EdgeToDOF(enr, col);
        rvs[l] = facetMat(tFOff + kRow, col);
      });
    });

    int presOff = tFOff + embRowsTF.Size();

    // cout << " facetMat presVec rows, " << facetMat.Height() - presOff << " = " << numPreserved << endl;
    // cout << facetMat.Rows(presOff, facetMat.Height()) << endl;
    // cout << endl;

    for (auto k : Range(numPreserved))
    {
      auto pVec = presVecs[k]->FVDouble();

      for (auto j : Range(edgeDofs.Size()))
      {
        auto const dof = edgeDofs.First() + j;

        pVec(dof) = facetMat(presOff + k, j);
      }
    }

  }, true); // NEED to compute facetMat


  return std::make_tuple(embProl, facetSurf, facetFlow, presVecs);
} // HDivHDGEmbedding::CreateDOFEmbeddingImpl


// shared_ptr<PreservedVectors>
// HDivHDGEmbedding::
// CreatePreservedVectors(BaseDOFMapStep const &E)
// {
//   throw Exception("I don't think we actually need HDivHDGEmbedding::CreatePreservedVectors!");
//   return nullptr;
// } // HDivHDGEmbedding::CreatePreservedVectors


template <ELEMENT_TYPE ET, int FACET_ORDER>
HDivFiniteElement<ET_trait<ET>::DIM>&
HDivHDGEmbedding::
T_GetHDivFE (int elnr, LocalHeap & lh) const
{
  Ngs_Element ngel = _hDivSpace->GetMeshAccess()->GetElement<ET_trait<ET>::DIM,VOL> (elnr);
  if (!_hDivSpace->DefinedOn(ngel)) return * new (lh) HDivDummyFE<ET>();

  HDivHighOrderFE<ET> * hofe =  new (lh) HDivHighOrderFE<ET> ();

  hofe->SetVertexNumbers (ngel.Vertices());
  hofe->SetHODivFree(false);
  hofe->SetOnlyHODiv(false);
  hofe->SetRT(false);

  constexpr int DIM     = ET_trait<ET>::DIM;
  constexpr int N_FACET = ET_trait<ET>::N_FACET;

  IVec<ET_trait<ET>::DIM> orderInner;
  orderInner = 0;

  IVec<N_FACET> orderFacet;
  orderFacet = FACET_ORDER;

  // IVec<N_FACET, IVec<DIM-1>> orderFacet;
  // Iterate<N_FACET>([&](auto i) { orderFacet[i] = 0; });

  hofe->SetOrderInner(orderInner); // don't need inner DOFs!
  hofe->SetOrderFacet(orderFacet); // only need lowest order DOF

  hofe->ComputeNDof();

  // cout << " T_GetHDivFE elnr " << elnr << ", ND " << hofe->GetNDof() << endl;
  return *hofe;
} // HDivHDGEmbedding::T_GetHDivFE


template<int DIM, int FACET_ORDER>
HDivFiniteElement<DIM>&
HDivHDGEmbedding::
GetHDivFE (ElementId ei, LocalHeap & lh) const
{
  if (ei.IsVolume())
  {
    int elnr = ei.Nr();
    Ngs_Element ngel = GetFESpace().GetMeshAccess()->GetElement(ei);
    ELEMENT_TYPE eltype = ngel.GetType();

    if constexpr( DIM == 2 )
    {
      switch (eltype)
      {
        case ET_TRIG:  { return T_GetHDivFE<ET_TRIG, FACET_ORDER>  (elnr, lh); }
        case ET_QUAD:  { return T_GetHDivFE<ET_QUAD, FACET_ORDER>  (elnr, lh); }
        default:
        {
          throw Exception ("illegal element in HDivHDGEmbedding::GetHDivFE");
        }
      }
    }
    else
    {
      switch (eltype)
      {
        case ET_TET:   { return T_GetHDivFE<ET_TET,   FACET_ORDER> (elnr, lh); }
        case ET_PRISM: { return T_GetHDivFE<ET_PRISM, FACET_ORDER> (elnr, lh); }
        case ET_HEX:   { return T_GetHDivFE<ET_HEX,   FACET_ORDER> (elnr, lh); }
        default:
        {
          throw Exception ("illegal element in HDivHDGEmbedding::GetHDivFE");
        }
      }
    }
  }

  throw Exception ("HDivHDGEmbedding::GetHDivFE only implemented for VOL elements!");

  if constexpr( DIM == 2 )
  {
    return * new (lh) HDivDummyFE<ET_TRIG>();
  }
  else
  {
    return * new (lh) HDivDummyFE<ET_TET>();
  }
} // HDivHDGEmbedding::GetHDivFE

/** END HDivHDGEmbedding **/

} // namespace amg
