function PackLogicalsTests
    % Packing tests.
    algset = (rand(100, 1000) > 0.6);
    assert(all(all(algset==UnpackLogicals(PackLogicals(algset), size(algset, 2)))));
    
    % Tests for AlgSet and SortedMatrix (including Create, Add, Contains)
    algset = GenerateRandomAlgset(100, 1000, 0.6);

    for iDel = 1:3
        for i=2:algset.Count
            alg1 = algset.Data(algset.Idx.Data(i-1), :);
            alg2 = algset.Data(algset.Idx.Data(i), :);
            assert(IsLexLess(alg1, alg2));
        end

        for i=1:algset.Count
            alg = algset.Data(algset.Idx.Data(i), :);
            assert(AlgsetContains(algset, alg));
            assert(AlgsetContains(algset, AlgsetGet(algset, i)));        
        end
        
        toDel = randi(algset.Count);
        algset = SortedMatrixRemove(algset, algset.Data(toDel, :));
    end
end