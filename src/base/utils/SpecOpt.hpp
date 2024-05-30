#ifndef FILE_SPECOPT_HPP
#define FILE_SPECOPT_HPP

#include <base.hpp>

namespace amg
{

/**
 * Options that have a configurable "default" value,
 * and optionally different "special" values for certain levels.
 * 
 * Configurable form Flags with syntax, special values with suffix "_spec".
 */ 
template<class OC>
class SpecOpt
{
public:
  OC default_opt;
  Array<OC> spec_opt;

  SpecOpt () : spec_opt(0) { ; }
  SpecOpt (OC _default_opt) : default_opt(_default_opt), spec_opt(0) { ; }
  SpecOpt (OC _default_opt, Array<OC> & _spec_opt) : default_opt(_default_opt), spec_opt(_spec_opt) { ; }
  SpecOpt (OC _default_opt, Array<OC> _spec_opt) : default_opt(_default_opt), spec_opt(_spec_opt) { ; }
  SpecOpt (const SpecOpt<OC> & other) : default_opt(default_opt), spec_opt(other.spec_opt) { ; }
  SpecOpt (SpecOpt<OC> && other) : default_opt(other.default_opt), spec_opt(move(other.spec_opt)) { ; }
  SpecOpt (const Flags & flags, OC defopt, string defkey, FlatArray<string> optkeys, FlatArray<OC> enum_vals)
    : SpecOpt(defopt)
  {
    SetFromFlagsEnum(flags, defkey, std::move(optkeys), std::move(enum_vals));
  }
  
  SpecOpt<OC> & operator = (OC defoc) {
    default_opt = defoc;
    spec_opt.SetSize0();
    return *this;
  }
  
  SpecOpt<OC> & operator = (const SpecOpt<OC> & other) {
    default_opt = other.default_opt;
    spec_opt.SetSize(other.spec_opt.Size());
    spec_opt = other.spec_opt;
    return *this;
  }

  void SetFromFlagsEnum (const Flags & flags, string defkey, FlatArray<string> optkeys, FlatArray<OC> enum_vals)
  {
    auto setoc = [&](auto & oc, string val) {
      int index;
      if ( (index = optkeys.Pos(val)) != -1)
        { oc = enum_vals[index]; }
    };
    setoc(default_opt, flags.GetStringFlag(defkey, ""));
    auto & specfa = flags.GetStringListFlag(defkey + "_spec");
    spec_opt.SetSize(specfa.Size()); spec_opt = default_opt;
    for (auto k : Range(spec_opt))
      { setoc(spec_opt[k], specfa[k]); }
  }

  void SetFromFlagsEnum (const Flags & flags, string defkey, Array<string>&& optkeys, Array<OC>&& enum_vals)
  { SetFromFlagsEnum(flags, defkey, FlatArray<string>(optkeys), FlatArray<OC>(enum_vals)); }
  INLINE void SetFromFlags (const Flags & flags, string key);
  OC GetOpt (int level) const { return (level < spec_opt.Size()) ? spec_opt[level] : default_opt; }

  SpecOpt<OC> & operator |= (const SpecOpt<OC> & other) {
    if constexpr(std::is_same<OC, bool>::value) {
      default_opt |= other.default_opt;
      for (auto k : Range(min(spec_opt.Size(), other.spec_opt.Size())))
        { spec_opt[k] |= other.spec_opt[k]; }
      if (spec_opt.Size() > other.spec_opt.Size())
        for (auto k : Range(other.spec_opt.Size(), spec_opt.Size()))
          { spec_opt[k] |= other.default_opt; }
    }
    return *this;
  }

  template<class X>
  friend INLINE std::ostream & operator<<(std::ostream &os, const SpecOpt<OC>& so);
}; // class SpecOpt

template<class OC>
INLINE void SpecOpt<OC> :: SetFromFlags (const Flags & flags, string key)
{
  throw Exception("SFF not overloaded!!");
} // SpecOpt::SetFromFlags

template<>
INLINE void SpecOpt<bool> :: SetFromFlags (const Flags & flags, string key)
{
  if (default_opt) { default_opt = !flags.GetDefineFlagX(key).IsFalse(); }
  else { default_opt = flags.GetDefineFlagX(key).IsTrue(); }
  auto & arr = flags.GetNumListFlag(key+"_spec");
  spec_opt.SetSize(arr.Size()); spec_opt = default_opt;
  for (auto k : Range(spec_opt))
    { spec_opt[k] = (arr[k] != 0); }
} // SpecOpt<bool>::SetFromFlags

template<>
INLINE void SpecOpt<xbool> :: SetFromFlags (const Flags & flags, string key)
{
  default_opt = flags.GetDefineFlagX(key);
  auto & arr = flags.GetNumListFlag(key+"_spec");
  spec_opt.SetSize(arr.Size()); spec_opt = default_opt;
  for (auto k : Range(spec_opt))
    { spec_opt[k] = (arr[k] == 0) ? false : true; }
} // SpecOpt<bool>::SetFromFlags

template<>
INLINE void SpecOpt<double> :: SetFromFlags (const Flags & flags, string key)
{
  default_opt = flags.GetNumFlag(key, default_opt);
  auto & arr = flags.GetNumListFlag(key+"_spec");
  spec_opt.SetSize(arr.Size()); spec_opt = default_opt;
  for (auto k : Range(spec_opt))
    { spec_opt[k] = arr[k]; }
} // SpecOpt<bool>::SetFromFlags


template<>
INLINE void SpecOpt<int> :: SetFromFlags (const Flags & flags, string key)
{
  default_opt = int(flags.GetNumFlag(key, double(default_opt)));
  auto & arr = flags.GetNumListFlag(key+"_spec");
  spec_opt.SetSize(arr.Size()); spec_opt = default_opt;
  for (auto k : Range(spec_opt))
    { spec_opt[k] = arr[k]; }
} // SpecOpt<bool>::SetFromFlags


template<>
INLINE void SpecOpt<string> :: SetFromFlags (const Flags & flags, string key)
{
  default_opt = flags.GetStringFlag(key, default_opt);
  auto & arr = flags.GetStringListFlag(key+"_spec");
  spec_opt.SetSize(arr.Size()); spec_opt = default_opt;
  for (auto k : Range(spec_opt))
    { spec_opt[k] = arr[k]; }
} // SpecOpt<bool>::SetFromFlags

template<typename X>
void SetEnumOpt (const Flags & flags, X & opt, string key, Array<string> vals, Array<X> evals) {
  string val = flags.GetStringFlag(key, "");
  for (auto k : Range(vals)) {
    if (val == vals[k])
      { opt = evals[k]; return; }
  }
}

template<typename X>
void SetEnumOpt (const Flags & flags, X & opt, string key, Array<string> vals, Array<X> evals, X default_val) {
  string val = flags.GetStringFlag(key, "");
  for (auto k : Range(vals)) {
    if (val == vals[k])
      { opt = evals[k]; return; }
  }
  opt = default_val;
}

template<class OC>
INLINE std::ostream & operator<<(std::ostream &os, const SpecOpt<OC>& so)
{
  os << "SpecOpts<" << typeid(OC).name() << ">, default val = [[" << so.default_opt << "]], spec opts (" << so.spec_opt.Size() << ") = ";
  prow2(so.spec_opt, os);
  return os;
}

} // namespace amg

#endif // FILE_SPECOPT_HPP
