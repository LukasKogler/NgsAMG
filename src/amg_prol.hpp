#ifndef FILE_AMG_PROL
#define FILE_AMG_PROL

namespace amg {

  class ProlongationClasses
  {
  public:
    struct ProlongationInfo
    {
      bool isparallel;
      string name;
      shared_ptr<BaseSparseMatrix>
      (*creator) ( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
		   const shared_ptr<const BaseAlgebraicMesh> & cmesh,
		   const shared_ptr<const CoarseMapping> & cmap);
    };

    Array<shared_ptr<ProlongationInfo>> pinfos;

    ProlongationClasses(){}
    ~ProlongationClasses(){}
    
    void AddProlongation (const string & name,
			  bool isparallel,
			  shared_ptr<BaseSparseMatrix>
			  (*creator) ( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
				       const shared_ptr<const BaseAlgebraicMesh> & cmesh,
				       const shared_ptr<const CoarseMapping> & cmap) ){}

    
    const Array<shared_ptr<ProlongationInfo>> & GetProlongations() { return pinfos; }
    const shared_ptr<ProlongationInfo> GetProlongation(const string & name) { return nullptr; }

    void PrintClasses(ostream & ost) const {
      ost << " ----- " << endl;
      ost << "Available Prolongations:" << endl;
      ost << " ----- " << endl;
      for(auto & info:pinfos)
	ost << "name: " << info->name << ", parallel? " << info->isparallel << endl;
      ost << " ----- " << endl;
    }
    
  };
  extern ProlongationClasses & GetProlongationClasses();


  class RegisterProlongation
  {
  public:
    RegisterProlongation (const string & name,
			  bool isparallel,
			  shared_ptr<BaseSparseMatrix>
			  (*creator) ( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
				       const shared_ptr<const BaseAlgebraicMesh> & cmesh,
				       const shared_ptr<const CoarseMapping> & cmap) )
    {
      GetProlongationClasses().AddProlongation(name, isparallel, creator);
    }
    
    ~RegisterProlongation(){}
  };

  
  
  shared_ptr<BaseSparseMatrix>
  BuildH1PWProlongation( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
			 const shared_ptr<const BaseAlgebraicMesh> & cmesh,
			 const shared_ptr<const CoarseMapping> & cmap);
  
  
  shared_ptr<BaseSparseMatrix>
  BuildH1HierarchicProlongation( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
				 const shared_ptr<const BaseAlgebraicMesh> & cmesh,
				 const shared_ptr<const CoarseMapping> & cmap);

  
  shared_ptr<BaseSparseMatrix>
  BuildlastiPWProlongation( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
			    const shared_ptr<const BaseAlgebraicMesh> & cmesh,
			    const shared_ptr<const CoarseMapping> & cmap,
			    Array<Vec<3,double> > & p_coords,
			    Array<Vec<3,double> > & cp_coords);

  
  shared_ptr<BaseSparseMatrix>
  BuildElastiHierarchicProlongation( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
				     const shared_ptr<const BaseAlgebraicMesh> & cmesh,
				     const shared_ptr<const CoarseMapping> & cmap,
				     Array<Vec<3,double> > & p_coords,
				     Array<Vec<3,double> > & cp_coords,
				     HashTable<INT<2>, INT<2,double>>* lamijs,
				     const shared_ptr<const BaseMatrix> & amat);

  shared_ptr<BaseSparseMatrix>
  BuildElastiHierarchicProlongation( const shared_ptr<const BaseAlgebraicMesh> & fmesh,
				     const shared_ptr<const BaseAlgebraicMesh> & cmesh,
				     const shared_ptr<const CoarseMapping> & cmap,
				     Array<Vec<3,double> > & p_coords,
				     Array<Vec<3,double> > & cp_coords,
				     HashTable<INT<2>, INT<4,double>>* lamijs,
				     const shared_ptr<const BaseMatrix> & amat);


  shared_ptr<BaseSparseMatrix>
  BuildVertexBasedHierarchicProlongation
  (const shared_ptr<const BaseAlgebraicMesh> & fmesh,
   const shared_ptr<const BaseAlgebraicMesh> & cmesh,
   const shared_ptr<const CoarseMapping> & cmap,
   const shared_ptr<const AMGOptions> & opts,
   const shared_ptr<SparseMatrix<double>> & pwp,
   size_t bsize, std::function<bool(const idedge&, FlatMatrix<double> & )> get_repl);


  /**
     This version computes values for collapsed vertices
     (that are free).
   **/
  shared_ptr<BaseSparseMatrix>
  BuildVertexBasedHierarchicProlongation2
  (const shared_ptr<const BaseAlgebraicMesh> & fmesh,
   const shared_ptr<const BaseAlgebraicMesh> & cmesh,
   const shared_ptr<const CoarseMapping> & cmap,
   const shared_ptr<const AMGOptions> & opts,
   const shared_ptr<SparseMatrix<double>> & pwp,
   const shared_ptr<BitArray> & freedofs,
   int BS, std::function<bool(const idedge&, FlatMatrix<double> & )> get_repl,
   std::function<double(size_t, size_t)> get_wt);

  // todo: clean this up
  shared_ptr<SparseMatrix<double>>
  HProl_Vertex
  (const shared_ptr<const BlockedAlgebraicMesh> & fmesh,
   Array<int> & vmap, //CoarseMap & cmap,
   const shared_ptr<ParallelDofs> & fpd, 
   const shared_ptr<ParallelDofs> & cpd, 
   const shared_ptr<const AMGOptions> & opts,
   const shared_ptr<SparseMatrix<double>> & pwp,
   const shared_ptr<BitArray> & freedofs,
   int BS, std::function<bool(const idedge&, FlatMatrix<double> & )> get_repl,
   std::function<double(size_t, size_t)> get_wt);

  
  
} // end namespace amg

#endif
