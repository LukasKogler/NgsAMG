#include <ngstd.hpp>
using namespace ngstd;


INLINE auto MyForward () { return tuple<>(); }
template <typename T1, typename...T> auto MyForward (T1 & a1, T&&...args);
template <typename T1, typename...T> auto MyForward (T1 && a1, T&&...args);

template <typename T1, typename...T>
auto MyForward (T1 & a1, T&&...args)
{
  return tuple_cat( tuple<T1&>(a1), MyForward(std::forward<T>(args)...) );
}

template <typename T1, typename...T>
auto MyForward (T1 && a1, T&&...args)
{
  return tuple_cat( tuple<T1>(a1), MyForward(std::forward<T>(args)...) );
}


template <typename TUP>
class cl_Zip
{
  TUP tup;

  template <typename TUP2>
  class MultipleIterator
  {
    TUP2 tup;
  public:
    MultipleIterator (TUP2 _tup)  : tup(_tup) { ; }
    MultipleIterator & operator++()
    {
      *this = apply([](auto...args) { return MultipleIterator(make_tuple(++args...)); }, tup);
      return *this;
    }
    
    bool operator!= (const MultipleIterator & it2)
    {
      return get<0> (tup) != get<0> (it2.tup);
    }
    
    auto operator*()
    {
      // return apply([](auto...args) { return make_tuple(*args...); }, tup);
      // return apply([](auto...args) { return forward_as_tuple(*args...); }, tup);
      return apply([](auto...args) { return MyForward(*args...); }, tup);
    }
  };
  
public:
  cl_Zip (TUP _tup) : tup(_tup) { ; }
  auto begin()
  {
    // return MultipleIterator(apply([](auto...cont) { return make_tuple(cont.begin()...); }, tup));
    auto begintup = apply([](auto&&...cont) { return make_tuple(cont.begin()...); }, tup);
    return MultipleIterator<decltype(begintup)>(begintup);
  }
  auto end()
  {
    // return MultipleIterator(apply([](auto...cont) { return make_tuple(cont.end()...); }, tup)); 
    auto endtup = apply([](auto&&...cont) { return make_tuple(cont.end()...); }, tup);
    return MultipleIterator<decltype(endtup)>(endtup);
  }
};

// template <typename T> T MakeView (T a) { return a; }
template <typename T> const T & MakeView (const T & a) { return a; }

template<typename T>
FlatArray<T> MakeView (const Array<T> & a) { return a; }


template <typename...T>
auto Zip (const T & ...args)
{
  return cl_Zip(make_tuple(MakeView(args)...));
}

template <typename T>
auto Enumerate (const T & arg)
{
  return Zip(Range(arg), arg);
}


template<class IT, class LAM>
class Map
{
private:
  IT it;
  LAM lam;
public:
  Map (IT && _it, LAM _lam) : it(move(_it)), lam(_lam) {}
  template<class INNER_IT>
  class Iterator
  {
    INNER_IT init;
    LAM lam;
  public:
    Iterator (INNER_IT _init, LAM _lam) : init(_init), lam(_lam) {}
    bool operator != (const Iterator<INNER_IT> & other) { return init!=other.init; }
    auto operator * () { return lam(*init); }
    Iterator & operator ++ () {++init; return *this; }
  };
  auto begin() { return Mapped_Iterator(it.begin(), lam); }
  auto end() { return Mapped_Iterator(it.end(), lam); }
};
