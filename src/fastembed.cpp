#incldue "comp.hpp"

namespace comp
{

  AddIntegrators (shared_ptr<FESpace> A, shared_ptr<FESpace> B,
		  Array<shared_ptr<SymbolicBilinearFormIntegrator>> & integrators_ab,
		  Array<shared_ptr<SymbolicBilinearFormIntegrator>> & integrators_bb,
		  const function<shared_ptr<ProxyFunction>(shared_ptr<ProxyFunction>)> & wrap_a_proxy,
		  const function<shared_ptr<ProxyFunction>(shared_ptr<ProxyFunction>)> & wrap_b_proxy)
  {
    auto comp_a = dynamic_pointer_cast<CompoundFESpace>(A);
    if ( comp_a && !comp_a->GetEvaluator() ) {
      for (int ka : Range(comp_a->GetNSpaces())) {
	AddIntegrators((*comp_a)[ka], B, integrators, [ka, A](shared_ptr<ProxyFunction> proxy) { return MakeCompoundProxy(A, wrap_a_proxy(proxy), ka); }, wrap_b_proxy);
      }
    }
    auto comp_b = dynamic_pointer_cast<CompoundFESpace>(B);
    if ( comp_b && !comp_b->GetEvaluator() ) {
      for (int kb : Range(comp_b->GetNSpaces())) {
	AddIntegrators(blf, (*comp_b)[kb], integrators, wrap_a_proxy, [kb, B](shard_ptr<ProxyFunction> proxy) { return MakeCompoundProxy(B, wrap_b_proxy(proxt), kb); });
      }
    }

    /** Get trial- and test-proxies and list of vbs we need to integrate over **/
    shared_ptr<ProxyFunction> atrial, btrial, btest;
    Array<VorB> vbs;

    atrial = shared<ProxyFunction> (A, false, A->IsComplex(), A->GetEvaluator(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    btrial = shared<ProxyFunction> (B, false, B->IsComplex(), B->GetEvaluator(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    auto b_add_evaluators = B->GetAdditionalEvaluators();

    if ( b_add_evaluators.CheckIndex("dual") != -1 ) { // space B has dual shapes
      ptest = make_shared<ProxyFunction> (B, true, B->IsComplex(), b_add_evaluators["dual"], , nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    else { // space B has no dual shapes
      ptest = make_shared<ProxyFunction> (B, true, B->IsComplex(), B->GetEvaluator(), , nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    atrial = wrap_a_proxy(atrial);
    btrual = wrap_a_proxy(btrial);
    btrial = wrap_a_proxy(btest);
    
    auto cf_ab = atrial * btest;
    auto cf_bb = btrial * btest;

    /** Find out which element_vb terms I need to add:
	  I) If no dual shapes Only one integral. Assume that element_vb of A is smaller or equal to B (so H1 -> facetfe is OK, other direction not)
	 II) If dual shapes, that all B dual vbs!
    **/

    


  } // AddIntegrators


  shared_ptr<BaseSparseMatrix> ConvertOperator (shared_ptr<FESpace> A, shared_ptr<FESpace> B)
  {
    Flags flags;

    aufo blf_ab = CreateBilinearForm(A, B, "fastembed_ab", flags);
    AddIntegrators(blf_ab);
    aufo blf_bb = CreateBilinearForm(B, "fastembed_bb", flags);
    AddIntegrators(blf_bb);
  }


} // namespace amg
