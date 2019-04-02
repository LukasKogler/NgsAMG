
#include "amg.hpp"

namespace amg
{

  void DOFMap :: SetCutoffs (FlatArray<size_t> acutoffs)
  {
    cutoffs.SetSize(acutoffs.Size());
    cutoffs = acutoffs;
    ConcSteps();
  }
    
  void DOFMap :: ConcSteps ()
  {
    cout << "CONCSTEPS: " << endl; prow2(steps); cout << endl;
    static Timer t("DOFMap::ConcSteps");
    RegionTimer rt(t);
    Array<shared_ptr<BaseDOFMapStep>> sub_steps(steps.Size()),
      new_steps(steps.Size()), rss(0);
    new_steps.SetSize(0);
    for (auto k : Range(cutoffs.Size())) {
      int next = cutoffs[k];
      if (k == 0) {
	if (next != 0) throw Exception("FIRST CUTOFF MUST BE 0!!");
	continue;
      }
      int last = cutoffs[k-1];
      if (next == last+1) {
	new_steps.Append(steps[last]);
      }
      else { // next >= last+2
	sub_steps.SetSize(0);
	int curr = next-1;
	shared_ptr<BaseDOFMapStep> cstep = steps[curr];
	cout << "take steps[" << curr << "]" << endl;
	curr--; bool need_one = false;
	while (curr >= last) {
	  cout << "try to conc with steps[" << curr << "]" << endl;
	  if ( auto cstep2 = steps[curr]->Concatenate(cstep) ) {
	    cout << "success!" << endl;
	    cstep = cstep2;
	    need_one = true;
	  }
	  else {
	    cout << "nope, append cstep" << endl;
	    sub_steps.Append(cstep);
	    cout << "take steps[" << curr << "]" << endl;
	    cstep = steps[curr];
	    need_one = true;
	  }
	  curr--;
	}
	if ( need_one ) {
	  cout << "append final cstep" << endl;
	  sub_steps.Append(cstep);
	}
	if (sub_steps.Size()==1)
	  {
	    cout << "just one step" << endl;
	    new_steps.Append(cstep);
	  }
	else {
	  cout << "composite step from " << sub_steps.Size() << endl;
	  rss.SetSize(sub_steps.Size());
	  for (auto l : Range(rss.Size()))
	    rss[l] = sub_steps[rss.Size()-1-l];
	  new_steps.Append(make_shared<ConcDMS>(rss));
	}
      }
    }
    steps = move(new_steps);
  }

  Array<shared_ptr<BaseSparseMatrix>> DOFMap :: AssembleMatrices (shared_ptr<BaseSparseMatrix> finest_mat) const
  {
    static Timer t("DOFMap::AssembleMatrices");
    RegionTimer rt(t);
    shared_ptr<BaseSparseMatrix> last_mat = finest_mat;
    Array<shared_ptr<BaseSparseMatrix>> mats;
    mats.Append(last_mat);
    // cout << "SS IS : " << steps.Size() << endl;
    for (auto step_nr : Range(steps.Size())) {
      cout << "do step " << step_nr << " of " << steps.Size() << endl;
      auto & step = steps[step_nr];
      auto next_mat = step->AssembleMatrix(last_mat);
      mats.Append(next_mat);
      last_mat = next_mat;
    }
    return mats;
  }

  
  
} // namespace amg
