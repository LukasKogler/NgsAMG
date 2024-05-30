#ifndef FILE_BASE_INCLUDES
#define FILE_BASE_INCLUDES

#include <comp.hpp>

/**
 * Compatibility with older NGSolve (delete this code when I pull NGSolve again)
*/
namespace ngcore
{
  template<> struct MPI_typetrait<float> {
    static MPI_Datatype MPIType () { return MPI_FLOAT; }
  };
}

#include <bla_extension.hpp>
#include <ng_mpi.hpp>


namespace ngbla

namespace amg {
  using namespace std;
  using namespace ngcomp;
  using namespace ngbla;

  /**
   *  HEIGHT and WIDTH have been removed from mat_traits in NGSolve,
   *  but Height<T>() does not work for Vec<K, double> since that class
   *  also has a non-static "Height" method, we have to use this workaround.
   *  I am adding this here because it will probably be needed everywhere.
   */
  template <class TM>
  INLINE constexpr size_t VecHeight () { return TM::HEIGHT; }

  template <>
  INLINE constexpr size_t VecHeight<double> () { return 1; }

  template <class TM>
  INLINE constexpr size_t VecWidth () { return TM::WIDTH; }

  template <>
  INLINE constexpr size_t VecWidth<double> () { return 1; }

  /**
   *  Similar story for general Height - cannot call Height<double>()
   */

  template <class TM>
  INLINE constexpr size_t TMHeight () { return TM::Height(); }

  template <>
  INLINE constexpr size_t TMHeight<double> () { return 1; }

  template <class TM>
  INLINE constexpr size_t TMWidth () { return TM::Width(); }

  template <>
  INLINE constexpr size_t TMWidth<double> () { return 1; }
} // namespace amg


// okay, I think those are the headers we ALWAYS want, they rarely change so it should be ok
#include <mpiwrap_extension.hpp>
#include <eqchierarchy.hpp>
#include <utils.hpp>

// #include "amg_typedefs.hpp"
// #include "amg_spmstuff.hpp"
// #include "amg_utils.hpp"  // needs to come after spmstuff
// #include "amg_bla.hpp" // by now, this is unfortunately used in a bunch of places
// #include "eqchierarchy.hpp"
// #include "reducetable.hpp"
// #include "amg_mesh.hpp"

namespace amg
{

template<class T> struct SIZE_IN_BUFFER_TRAIT { };

} // namespace amg

#endif // FILE_BASE_INCLUDES