#ifdef STOKES

#ifndef FILE_AMG_PC_STOKES_IMPL_HPP
#define FILE_AMG_PC_STOKES_IMPL_HPP

namespace amg
{

  /** StokesAMGPC **/
  
  template<class FACTORY, class AUX_SYS>
  StokesAMGPC<FACTORY, AUX_SYS> :: StokesAMGPC (const PDE & apde, const Flags & aflags, const string aname)
    : BaseAMGPC(apde, aflags, aname)
  { throw Exception("PDE-Constructor not implemented!"); }

  template<class FACTORY, class AUX_SYS>
  StokesAMGPC<FACTORY, AUX_SYS> :: StokesAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts);
    : BaseAMGPC(apde, aflags, aname)
  {
    aux_sys = make_shared<AUX_SYS>(blf->GetFESpace());
  } // StokesAMGPC(..)

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: InitLevel (shared_ptr<BitArray> freedofs)
  {
    /** Initialize auxiliary system (filter freedofs / auxiliary freedofs, pardofs / convert-operator / alloc auxiliary matrix ) **/
    aux_sts->Initialize(freedofs);
  } // StokesAMGPC::InitLevel
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseAMGPC::Options> StokesAMGPC<FACTORY, AUX_SYS> :: NewOpts ()
  {
    
  } // StokesAMGPC::NewOpts
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: SetDefaultOptions (BaseAMGPC::Options& O)
  {
  } // StokesAMGPC::SetDefaultOptions
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: SetOptionsFromFlags (BaseAMGPC::Options& O, const Flags & flags, string prefix)
  {
  } // StokesAMGPC::SetOptionsFromFlags
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: ModifyOptions (BaseAMGPC::Options & O, const Flags & flags, string prefix)
  {
  } // StokesAMGPC::ModifyOptions
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<TopologicMesh> StokesAMGPC<FACTORY, AUX_SYS> :: BuildInitialMesh ()
  {
    static Timer t("BuildInitialMesh");
    RegionTimer rt(t);
    auto eqc_h = BuildEQCH();
    return BuildAlgMesh(BuildTopMesh(eqc_h));
  } // StokesAMGPC::BuildInitialMesh
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
  {
  } // StokesAMGPC::InitFinestLevel
  

  template<class FACTORY, class AUX_SYS>
  Table<int> StokesAMGPC<FACTORY, AUX_SYS> :: GetGSBlocks (const BaseAMGFactory::AMGLevel & amg_level)
  {
  } // StokesAMGPC::GetGSBlocks
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseAMGFactory> StokesAMGPC<FACTORY, AUX_SYS> :: BuildFactory ()
  {
  } // StokesAMGPC::BuildFactory
  

  template<class FACTORY, class AUX_SYS>
  shared_ptr<BaseDOFMapStep> StokesAMGPC<FACTORY, AUX_SYS> :: BuildEmbedding (shared_ptr<TopologicMesh> mesh)
  {
  } // StokesAMGPC::BuildEmbedding
  

  template<class FACTORY, class AUX_SYS>
  void StokesAMGPC<FACTORY, AUX_SYS> :: RegularizeMatrix (shared_ptr<BaseSparseMatrix> mat, shared_ptr<ParallelDofs> & pardofs) const
  {
  } // StokesAMGPC::RegularizeMatrix


  template<class FACTORY, class AUX_SYS>
  shared_ptr<EQCHierarchy> StokesAMGPC<FACTORY, AUX_SYS> :: BuildEQCH ();
  {
  } // StokesAMGPC::BuildEQCH


  template<class FACTORY, class AUX_SYS>
  shared_ptr<BlockTM> StokesAMGPC<FACTORY, AUX_SYS> :: BuildTopMesh (shared_ptr<EQCHierarchy> eqc_h);
  {
    /** VOL-EL -> vertex
	FACET -> edge **/
    

  } // StokesAMGPC::BuildTopMesh


  template<class FACTORY, class AUX_SYS>
  shared_ptr<TMESH> StokesAMGPC<FACTORY, AUX_SYS> :: BuildAlgMesh (shared_ptr<BlockTM> top_mesh);
  {
  } // StokesAMGPC::BuildAlgMesh


  /** END StokesAMGPC **/

} // namespace amg

#endif // FILE_AMG_PC_STOKES_IMPL_HPP
#endif // STOKES
