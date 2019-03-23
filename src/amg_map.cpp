
#include "amg.hpp"

namespace amg
{
  
  Array<shared_ptr<BaseSparseMatrix>> DOFMap :: AssembleMatrices (shared_ptr<BaseSparseMatrix> finest_mat) const
  {
    auto last_mat = finest_mat;
    Array<shared_ptr<BaseSparseMatrix>> mats;
    mats.Append(last_mat);
    // cout << "SS IS : " << steps.Size() << endl;
    for (auto step_nr : Range(steps.Size())) {
      // cout << "do step " << step_nr << " of " << steps.Size() << endl;
      if (step_nr==steps.Size()) { /* cout << "wtf.." << endl; */ break; }
      auto & step = steps[step_nr];
      auto next_mat = step->AssembleMatrix(last_mat);
      cout << "ok!!, next mat: " << endl << next_mat << endl;
      if (next_mat!=nullptr) cout << *next_mat << endl;
      else cout << "nullptr!!!" << endl;
      mats.Append(next_mat);
      last_mat = next_mat;
    }
    return mats;
  }

} // namespace amg
