function isInternalClosure = IsInternalClosure(algset, edges)
    nAlgs = algset.Count;
    for i = 1:nAlgs
        alg = AlgsetGet(algset, i);
        children = GetChildren(edges, i);
        for child1 = children
            child1alg = AlgsetGet(algset, child1);
            for child2 = children
                if (child1 < child2)
                    child2alg = AlgsetGet(algset, child2);
                    if any((child1alg & child2alg) ~= alg)
                        isInternalClosure = false;
                        return;
                    end
                end
            end
        end        
    end
    
    isInternalClosure = true;
end