#ifndef FILE_MPIWRAP_EXTENSION
#define FILE_MPIWRAP_EXTENSION

#include "amg.hpp"

#include <ngstd.hpp>

namespace amg
{

  enum { MPI_TAG_AMG = 1120 };

  class BlockTM;
  
  class NgsAMG_Comm : public NgsMPI_Comm
  {
  public:
    NgsAMG_Comm (const NgsMPI_Comm & c) : NgsMPI_Comm(c) { ; }
    NgsAMG_Comm (NgsMPI_Comm && c) : NgsMPI_Comm(c) { ; }
    NgsAMG_Comm (MPI_Comm comm, bool owns)
      : NgsMPI_Comm(NgMPI_Comm(comm, owns)) { ; }
    NgsAMG_Comm () : NgsAMG_Comm(NgsMPI_Comm(NgMPI_Comm())) { ; }
    ~NgsAMG_Comm () { ; }

    using NgsMPI_Comm :: Send;
    using NgsMPI_Comm :: Recv;

  private:
    INLINE Timer& thack_send_tab () const { static Timer t("Send Table"); return t; }
  public:
    template<typename T, typename T2 = decltype(GetMPIType<T>())>
    void Send (FlatTable<T> tab, int dest, int tag) const {
      RegionTimer rt(thack_send_tab());
      size_t size = tab.Size();
      Send(size, dest, tag);
      if(!size) return;
      Send(tab.IndexArray(), dest, tag);
      Send(tab.AsArray(), dest, tag);
      return;
    }
    
  private:
    INLINE Timer& thack_recv_tab () const { static Timer t("Recv Table"); return t; }
  public:
    template<typename T, typename T2 = decltype(GetMPIType<T>())>
    void Recv (Table<T> & tab, int src, int tag) const {
      RegionTimer rt(thack_recv_tab());
      size_t size = -1;
      Recv(size, src, tag);
      if(!size) {      
	tab = Table<T>();
	return;
      }
      Array<size_t> index(size+1);
      Recv(index, src, tag);
      Array<int> sizes(size);
      for(auto k:Range(size))
	sizes[k] = index[k+1]-index[k];
      tab = Table<T>(sizes);
      Recv(tab.AsArray(), src, tag);
    }

  private:
    INLINE Timer& thack_bcast_fa () const { static Timer t("Bcast Array"); return t; }
  public:
    template<typename T, typename T2 = decltype(GetMPIType<T>())>
    void Bcast (FlatArray<T> ar, int root) const {
      RegionTimer(thack_bcast_fa());
      if (ar.Size() == 0) return;
      MPI_Bcast (&ar[0], ar.Size(), GetMPIType<T>(), root, comm);
    }
    
  private:
    INLINE Timer& thack_bcast_tab () const { static Timer t("Bcast Table"); return t; }
  public:
    template<typename T, typename T2 = decltype(GetMPIType<T>())>
    void Bcast (Table<T> & tab, int root) const {
      RegionTimer rt(thack_bcast_tab());
      INT<2, size_t> ss = {tab.Size(), (tab.Size() ? tab.AsArray().Size() : 0)};
      // MyMPI_Bcast(ss, *this, root);
      NgMPI_Comm::Bcast(ss, root);
      if (Rank() != root) {
	Array<int> perow(ss[0]);
	perow = 0; perow.Last() = ss[1];
	tab = Table<T>(perow);
      }
      if (ss[1]>0) {
	Bcast(tab.IndexArray(), root);
	Bcast(tab.AsArray(), root);
      }
    }
    
  private:
    INLINE Timer& thack_isend_spm () const { static Timer t("ISend SPM"); return t; }
  public:
    template<typename T, typename T2 = decltype(GetMPIType<T>())>
    MPI_Request ISend (const ngla::SparseMatrix<T> & spm, int dest, int tag) const {
      RegionTimer rt(thack_isend_spm());
      size_t H = spm.Height(); size_t W = spm.Width();
      if (H!=W) throw Exception("cant rly deal with H!=W ISend"); // need to send W, need temporary mem
      MPI_Request req = MyMPI_ISend (spm.GetFirstArray(), dest, tag, comm);
      size_t nzes = spm.GetFirstArray().Last();
      if (H>0 && W>0) {
	MPI_Request_free(&req);
	int* cp = &spm.GetRowIndices(0)[0];
	FlatArray<int> cols(nzes, cp);
	req = MyMPI_ISend (cols, dest, tag, comm);
	MPI_Request_free(&req);
	T* vp = &spm.GetRowValues(0)[0];
	FlatArray<T> vals(nzes, vp);
	req = MyMPI_ISend (vals, dest, tag, comm);
	return req;
      }
      return req;
    }

  private:
    INLINE Timer& thack_send_spm () const { static Timer t("Send SPM"); return t; }
  public:
    template<typename T, typename T2 = decltype(GetMPIType<T>())>
    void Send (const ngla::SparseMatrix<T> & spm, int dest, int tag) const {
      RegionTimer rt(thack_send_spm());
      MPI_Request req = ISend(spm, dest, tag); // well this is just lazy ...
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

  private:
    INLINE Timer& thack_recv_spm () const { static Timer t("Recv SPM"); return t; }
  public:
    template<typename T, typename T2 = decltype(GetMPIType<T>())>
    void Recv (shared_ptr<ngla::SparseMatrix<T> >& spm, int src, int tag) const {
      RegionTimer rt(thack_recv_spm());
      Array<size_t> firstia;
      Recv(firstia, src, tag);
      size_t H = firstia.Size()-1; size_t W = H;
      Array<int> nperow(H);
      for (auto k:Range(H)) nperow[k] = firstia[k+1]-firstia[k];
      spm = make_shared<ngla::SparseMatrix<T>>(nperow, W);
      if (H==0 || W==0) return;
      int* cp = &spm->GetRowIndices(0)[0];
      size_t nzes = 0; for (auto k:Range(H)) nzes += nperow[k];
      FlatArray<int> cols(nzes, cp);
      Recv (cols, src, tag);
      T* vp = &spm->GetRowValues(0)[0];
      FlatArray<T> vals(nzes, vp);
      Recv (vals, src, tag);
      // cout << "GOT SPM: " << endl << *spm << endl;
    }

    void Send (shared_ptr<BlockTM> & mesh, int dest, int tag) const;
    void Recv (shared_ptr<BlockTM> & mesh, int src,  int tag) const;
    
  }; // class NgsAMG_Comm
  
} // namespace amg

namespace ngstd
{

  template<typename T>
  inline MPI_Request MyMPI_ISend (const FlatTable<T> tab, int dest, int tag, NgsMPI_Comm comm)
  {
    MPI_Request req;
    size_t size = tab.Size();
    req = MyMPI_ISend(size, dest, tag, comm);
    if (!size) return req;
    MPI_Request_free(&req);
    req = MyMPI_ISend(tab.IndexArray(), dest, tag, comm);
    MPI_Request_free(&req);
    req = MyMPI_ISend(tab.AsArray(), dest, tag, comm);
    return req;
  }
  
  // this doesnt work!!
  template<typename T>
  inline MPI_Request MyMPI_ISend(const ngla::SparseMatrix<T> & spm, int dest, int tag, NgsMPI_Comm comm)
  {
    MPI_Request req;
    /** send height, width & NZE **/
    size_t h = spm.Height();
    size_t w = spm.Width();
    size_t nze = spm.NZE();
    const auto sym_spm = dynamic_cast<const ngla::SparseMatrixSymmetric<T>*>(&spm);
    size_t is_sym = (sym_spm!=NULL)?1:0;
    INT<4, size_t> data({h, w, nze, is_sym});
    req = MyMPI_ISend(data, dest, tag, comm);
    MPI_Request_free(&req);
    /** send row-indices **/
    req = MyMPI_ISend(spm.firsti, dest, tag, comm);
    MPI_Request_free(&req);
    /** send col-nrs & vals **/
    int* cptr = NULL;
    T* vptr = NULL;
    for(size_t k=0;k<h && cptr==NULL; k++)
      if(spm.GetRowIndices(k).Size()) {
	cptr = &spm.GetRowIndices(k)[0];
	vptr = &spm.GetRowValues(k)[0];
      }
    req = MyMPI_ISend(FlatArray<int>(nze, cptr), dest, tag, comm);
    MPI_Request_free(&req);
    req = MyMPI_ISend(FlatArray<T>(nze, vptr), dest, tag, comm);
    return req;
  }
  
} // namespace ngstd

#endif
