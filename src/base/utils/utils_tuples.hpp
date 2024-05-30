#include <base.hpp>

namespace amg
{

template<class TLAM, class... TX>
INLINE void ApplyComponentWise(TLAM const &lam, tuple<TX*...> &x)
{
 Iterate<count_ppack<TX...>::value>( [&](auto i) {
    lam(get<i.value>(x));
  } );
}

template<class TLAM, class... TX>
INLINE void ApplyComponentWise(TLAM const &lam, tuple<TX*...> const &x)
{
 Iterate<count_ppack<TX...>::value>( [&](auto i) {
    lam(get<i.value>(x));
  } );
}

template<class TLAM, class... TX, class... TY>
INLINE void ApplyComponentWise(TLAM const &lam, tuple<TX*...> &x, tuple<TY*...> &y)
{
 Iterate<count_ppack<TX...>::value>( [&](auto i) {
    lam(get<i.value>(x), get<i.value>(y));
  } );
}

template<class TLAM, class... TX, class... TY>
INLINE void ApplyComponentWise(TLAM const &lam, tuple<TX*...> const &x, tuple<TY*...> const &y)
{
 Iterate<count_ppack<TX...>::value>( [&](auto i) {
    lam(get<i.value>(x), get<i.value>(y));
  } );
}

} // namespace amg
