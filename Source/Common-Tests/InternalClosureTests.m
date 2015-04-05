function InternalClosureTests
    algset = GenerateRandomAlgset(150, 20, 0.5);
    edges = BuildHasseGraph(algset);
	
    % Test for internal closure
    [algset_IC, edges_IC] = BuildInternalClosure(algset, edges);
    assert(IsInternalClosure(algset_IC, edges_IC));
    for i = 1:algset.Count
        assert(AlgsetContains(algset_IC, AlgsetGet(algset, i)));
    end

    % SAU Tests
    v = [32 16 8 4 2 1];
    sau_storage = CreateSAU(length(v));
    acc_sau = [];
    for i = 0:sum(v)
        [sau, sau_storage] = CalcSAU([i v], sau_storage);
        acc_sau = [acc_sau, sau];
    end
    assert(sum(acc_sau) == prod(v+1));  
end
