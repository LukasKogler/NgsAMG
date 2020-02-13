#ifndef FILE_FASTEMBED_HPP
#define FILE_FASTEMBED_HPP 

namespace amg
{

  /** An AMG preconditioner in the auxiliary space  SPACE.
      Embedding via dualshapes of VECSPACE. **/
  template<int BS, class SPACE, class VECSPACE>
  class AuxiliarySpaceAMG
  {
  protected:

    /** Embedding: multidim -(Mat<BS, 1>)-> vec -(dualshapes)-> Xlo -(lo-to-ho embedding)-> X **/
    shared_ptr<SPACE> Vmd;
    shared_ptr<VECSPACE> Vvec;
    shared_ptr<CompoundFESpace> X, Xlo;
    Array<int> inds;
    Table<bool> use_vb; // use_vb[i][VB] = does Xlo.components[i] have dual shapes for VB?

    shared_ptr<BilinearForm> mVX, mXX;
    
    /** PC in Vmd **/
    shared_ptr<BaseAMGPC> Vpc;
    
    /** Smoother in X **/
    NODE_TYPE bs_block_nodes;

    shared_ptr<EmbeddedAMGMatrix> emb_amg_mat;

  public:

    virtual shared_ptr<EmbeddedAMGMatrix> GetMatrix() { return emb_amg_mat; }
  }; // class AuxiliarySpaceAMG

} // namespace amg

#endif
