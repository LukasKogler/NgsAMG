#ifndef FILE_PLATE_TEST_AGG_HPP
#define FILE_PLATE_TEST_AGG_HPP

#include <utils_numeric_types.hpp>

#include "agglomerator.hpp"

namespace amg
{

/** Puts everything in z-direction into the same agg **/

template<class ATMESH>
class PlateTestAgglomerator : public Agglomerator<ATMESH>
{
  static constexpr bool ROBUST = true;
  using TMESH = ATMESH;

public:
  PlateTestAgglomerator (shared_ptr<TMESH> _mesh)
    : Agglomerator<TMESH>(_mesh)
  {
    ;
  }

  void Initialize (const AggOptions & opts, int level) { ; }

protected:
  virtual void FormAgglomerates (Array<Agglomerate> & agglomerates, Array<int> & v_to_agg) override;
}; // class PlateTestAgglomerator


} // namespace amg

#endif // FILE_PLATE_TEST_AGG_HPP
