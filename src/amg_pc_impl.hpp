#ifndef FILE_AMGPC_IMPL_HPP
#define FILE_AMGPC_IMPL_HPP

namespace amg
{

  /** Options **/

  struct BaseAMGPC :: Options
  {
    /** What we do on the coarsest level **/
    enum CLEVEL : char { INV_CLEV = 0,        // invert coarsest level
			 SMOOTH_CLEV = 1,    // smooth coarsest level
			 NO_CLEV = 2 };
    CLEVEL clev = INV_CLEV;
    INVERSETYPE cinv_type = MASTERINVERSE;
    INVERSETYPE cinv_type_loc = SPARSECHOLESKY;
    size_t clev_nsteps = 1;                   // if smoothing, how many steps do we do?
    
    /** Smoothers **/

    enum SM_TYPE : char /** available smoothers **/
      { GS = 0,     // (l1/hybrid - ) Gauss-Seidel
	BGS = 1 };  // Block - (l1/hybrid - ) Gauss-Seidel 
    SM_TYPE sm_type = SM_TYPE::GS;

    enum GS_VER : char /** different hybrid GS versions (mostly for testing) **/
      { VER1 = 0,    // old version
	VER2 = 1,    // newer (maybe a bit faster than ver3 without overlap)
	VER3 = 2 };  // newest, optional overlap
    GS_VER gs_ver = GS_VER::VER3;

    bool sm_mpi_overlap = true;          // overlap communication/computation (only VER3)
    bool sm_mpi_thread = false;          // do MPI-comm in seperate thread (only VER3)

    /** Misc **/
    bool sync = false;                   // synchronize via MPI-Barrier in places
    bool do_test = false;                // perform PC-test for amg_mat
  }; // BaseAMGPC::Options

  /** END Options **/


  /** BaseAMGPC **/

  BaseAMGPC :: BaseAMGPC (const PDE & apde, const Flags & aflags, const string aname)
    : Preconditioner(&apde, aflags, aname)
  { throw Exception("PDE constructor not implemented!"); }
    

  BaseAMGPC :: BaseAMGPC (shared_ptr<BilinearForm> blf, const Flags & flags, const string name, shared_ptr<Options> opts)
    : Preconditioner(bfa, flags, name), options(opts), bfa(bfa)
  {
    if (otps == nullptr)
      { opts = MakeOptionsFromFlags(flags); }
  } // BaseAMGPC(..)


  BaseAMGPC :: ~BaseAMGPC ()
  {
    ;
  } // ~BaseAMGPC


  void BaseAMGPC :: InitLevel (shared_ptr<BitArray> freedofs)
  {
  } // BaseAMGPC::InitLevel


  void BaseAMGPC :: FinalizeLevel (const BaseMatrix * mat)
  {
    if (mat != nullptr)
      { finest_mat = shared_ptr<BaseMatrix>(const_cast<BaseMatrix*>(mat), NOOP_Deleter); }
    else
      { finest_mat = bfa->GetMatrixPtr(); }

    Finalize();
  } // BaseAMGPC::FinalizeLevel


  const BaseMatrix & BaseAMGPC :: GetAMatrix () const 
  {
    if (finest_mat == nullptr)
      { throw Exception("BaseAMGPC - finest mat not ready!"); }
    return *finest_mat;
  } // BaseAMGPC


  const BaseMatrix & BaseAMGPC :: GetMatrix () const 
  {
    return *GetMatrixPtr();
  } // BaseAMGPC::GetMatrix


  shared_ptr<BaseMatrix> BaseAMGPC :: GetMatrixPtr () 
  {
    if (amg_mat == nullptr)
      { throw Exception("BaseAMGPC - amg_mat not ready!"); }
    return amg_mat;
  } // BaseAMGPC::GetMatrixPtr


  void BaseAMGPC :: Mult (const BaseVector & b, BaseVector & x) const 
  {
    GetMatrixptr()->Mult(b, x);
  } // BaseAMGPC::Mult


  void BaseAMGPC :: MultTrans (const BaseVector & b, BaseVector & x) const 
  {
    GetMatrixPtr()->MultTrans(b, x);
  } // BaseAMGPC::MultTrans


  void BaseAMGPC :: MultAdd (double s, const BaseVector & b, BaseVector & x) const 
  {
    GetMatrixPtr()->MultAdd(s, b, x);
  } // BaseAMGPC::MultAdd


  void BaseAMGPC :: MultTransAdd (double s, const BaseVector & b, BaseVector & x) const 
  {
    GetMatrixPtr()->MultTransAdd(s, b, x);
  } // BaseAMGPC::MultTransAdd


  shared_ptr<Options> BaseAMGPC :: MakeOptionsFromFlags (const Flags & flags, string prefix)
  {
    auto opts = NewOpts();
    SetDefaultOptions(*opts);
    SetOptionsFromFlags(*opts, flags, prefix);
    ModifyOptions(*opts, flags, prefix);
  } // BaseAMGPC::MakeOptionsFromFlags


  void BaseAMGPC :: SetOptionsFromFlags (Options& O, const Flags & flags, string prefix)
  {
    auto set_enum_opt = [&] (auto & opt, string key, Array<string> vals, auto default_val) {
      string val = flags.GetStringFlag(prefix + key, "");
      bool found = false;
      for (auto k : Range(vals)) {
	if (val == vals[k]) {
	  found = true;
	  opt = decltype(opt)(k);
	  break;
	}
      }
      if (!found)
	{ opt = default_val; }
    };

    auto set_bool = [&](auto& v, string key) {
      if (v) { v = !flags.GetDefineFlagX(prefix + key).IsFalse(); }
      else { v = flags.GetDefineFlagX(prefix + key).IsTrue(); }
    };

    auto set_opt_kv = [&](auto & opt, string name, Array<string> keys, auto vals) {
      string flag_opt = flags.GetStringFlag(pfit(name), "");
      for (auto k : Range(keys))
	if (flag_opt == keys[k]) {
	  opt = vals[k];
	  return;
	}
    };

    set_enum_opt(O.clev, "clev", {"inv", "sm", "none"}, Options::CLEVEL::INV_CLEV);

    set_opt_kv(O.cinv_type, "cinv_type", { "masterinverse", "mumps" }, Array<INVERSETYPE>({ MASTERINVERSE, MUMPS }));

    set_opt_kv(O.cinv_type_loc, "cinv_type_loc", { "pardiso", "pardisospd", "sparsecholesky", "superlu", "superlu_dist", "mumps", "umfpack" },
	       Array<INVERSETYPE>({ PARDISO, PARDISOSPD, SPARSECHOLESKY, SUPERLU, SUPERLU_DIST, MUMPS, UMFPACK }));

    set_enum_opt(O.sm_type, "sm_type", {"gs", "bgs"}, Options::SM_TYPE::GS);

    O.gs_ver = Options::GS_VER(max(0, min(3, int(flags.GetNumFlag(pfit("gs_ver"), 3)))));

    set_bool(O.sm_mpi_overlap, "sm_mpi_overlap");
    set_bool(O.sm_mpi_thread, "sm_mpi_thread");

    set_bool(O.sync, "sync");
    set_bool(O.do_test, "do_test");
  } //BaseAMGPC::SetOptionsFromFlags


  void BaseAMGPC :: SetDefaultOptions (Options& O)
  {
    ;
  } // BaseAMGPC::SetDefaultOptions


  void BaseAMGPC :: ModifyOptions (Options & O)
  {
    ;
  } // BaseAMGPC::ModifyOptions



  void BaseAMGPC :: FinalizeLevel (const BaseMatrix * mat)
  {
    finest_mat = mat;

    Finalize();
  } // BaseAMGPC::FinalizeLevel


  void BaseAMGPC :: Finalize ()
  {

    if (options->sync)
      {
	if (auto pds = finest_mat->GetParallelDofs()) {
	  static Timer t(string("Sync1")); RegionTimer rt(t);
	  pds->GetCommunicator().Barrier();
	}
      }

    if (finest_freedofs == nullptr)
      { finest_freedofs = bfa->GetFESpace()->GetFreeDofs(bfa->UsesEliminateInternal()); }
    
    /** Set dummy-ParallelDofs **/
    shared_ptr<BaseMatrix> fine_spm = finest_mat;
    if (auto pmat = dynamic_pointer_cast<ParallelMatrix>(fine_spm))
      { fine_spm = pmat->GetMatrix(); }
    else {
      Array<int> perow (fine_spm->Height() ); perow = 0;
      Table<int> dps (perow);
      NgsMPI_Comm c(MPI_COMM_WORLD);
      MPI_Comm mecomm = (c.Size() == 1) ? MPI_COMM_WORLD : AMG_ME_COMM;
      fine_spm->SetParallelDofs(make_shared<ParallelDofs> ( mecomm , move(dps), GetEntryDim(fine_spm.get()), false));
    }

    auto mesh = BuildInitialMesh();

    factory = BuildFactory(mesh);

    BuildAMGMat();
  } // BaseAMGPC::Finalize


  void BaseAMGPC :: BuildAMGMat ()
  {
    static Timer t("BuildAMGMat"); RegionTimer rt(t);

    auto dof_map = make_shared<DOFMap>();

    Array<BaseAMGFactory::AMGLevel> amg_levels(1); InitFinestLevel(amg_levels[0]);

    factory->SetupLevels(amg_levels, dof_map);

    Array<shared_ptr<BaseSmoother>> smoothers(amg_levels.Size() - 1);
    for (auto k : Range(amg_levels))
      { smoothers[k] = BuildSmoother(amg_levels[k]); }

  } // BaseAMGPC::BuildAMGMAt


  void BaseAMGPC :: InitFinestLevel (BaseAMGFactory::AMGLevel & finest_level)
  {
    finest_level.level = 0;
    finest_level.mesh = finest_mesh; // TODO: get out of factory??
    auto fpm = dynamic_pointer_cast<ParallelMatrix>(finest_mat)->GetMatrix();
    finest_level.pardofs = fpm->GetParallelDofs();
    finest_level.mat = dynamic_pointer_cast<BaseSparseMatrix>(fpm);
    finest_level.embed_map = BuildEmbedding();
  } // BaseAMGPC::InitFinestLevel


  /** END BaseAMGPC **/


  // /** EmbedVAMG **/

  // template<int N> struct has_smoother { static bool val = false; };
  // template<> struct has_smoother<1> { static bool val = true; };
  // template<> struct has_smoother<2> { static bool val = true; };
  // template<> struct has_smoother<3> { static bool val = true; };
  // template<> struct has_smoother<6> { static bool val = true; };
    
  // shared_ptr<BaseSmoother> EmbedVAMG :: BuildSmoother (const AMGLevel & amg_level)
  // {
  //   SM_TYPE sm_type = (O.spec_smt[amg_level.level] == SM_DEFAULT) ? O.def_sm : O.spec_smt[amg_level.level];

  //   shared_ptr<BaseSmoother> smoother;

  //   if (!( (amg_level.mat != nullptr)  && (amg_level.pardofs != nullptr) ) )
  //     { throw Exception("Cannot build smoother!"); }

  //   Iterate<MAX_SYS_DIM> ( [&](auto N) {
  // 	if (has_smoother<N.value>::val) {
  // 	  if ( auto spm = dynamic_pointer_cast<stripped_spm<Mat<N,N,double>>>(amg_level.mat) )
  // 	    switch(sm_type) {
  // 	    case(SM_GS) : { smoother = BuildGSS<N.level>(amg_level); }
  // 	    case(SM_BGS) : { smoother = BuildBGSS(amg_level); }
  // 	    default : { throw Exception("Unknown smoothing type!"); }
  // 	    }
  // 	}
  //     } );

  //   if (smoother == nullptr)
  //     { throw Exception(string("Could not build smoother from mat type = ") + typeid(*amg_level.mat).name()); }

  //   return smoother;
  // } // EmbedVAMG::BuildSmoother


  // template<int N>
  // shared_ptr<BaseSmoother> EmbedVAMG :: BuildVBlockSmoother (const AMGLevel & amg_level)
  // {
  //   auto spmat = dynamic_pointer_cast<SparseMatrix<Mat<N, N, double>>>(amg_level.mat);

  //   if (spmat == nullptr)
  //     { throw Exception("wrong mat type for smoother"); }

  //   shared_ptr<BaseSmoother> smoother;

  //   if (amg_level.level == 0) {
  //     /** Have to take into account finest_freedofs AND embedding mat! **/
  //   }
  //   else {
  //     /** All DOFs are free. Take crs-map**/
  //   }

  //   Table<int> c2fv = amg_level.crs_map->MoveReverseMap<NT_VERTEX>();
  //   if ( (amg_level.level == 0) && (amg_level.emb_map != nullptr) ) {
  //     /** **/
  //   }
  //   else {
  //     /** blocks are straightforward **/
  //   }

  //   return smoother;
  // } // EmbedVAMG::BuildSmoother

  // /** EmbedVAMG **/

} // namespace amg

#endif
