#ifndef FILE_AMG_SM_IMPL
#define FILE_AMG_SM_IMPL

namespace amg
{

  template<class TM>
  template<class TLAM>
  INLINE void GSS4<TM> :: iterate_rows (TLAM lam, bool bw) const
  {
    if (bw) {
      for (int rownr = cA->Height() - 1; rownr >= 0; rownr--)
	{ lam(rownr); }
    }
    else {
      for (auto k : Range(cA->Height()))
	{ lam(k); }
    }
  } // GSS4::iterate_rows


} // namespace amg

#endif // FILE_AMG_SM_IMPL
