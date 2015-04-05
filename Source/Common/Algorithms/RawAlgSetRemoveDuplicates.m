function rawalgset = RawAlgSetRemoveDuplicates(rawalgset)
    nAlgs = size(rawalgset, 1);
    rawalgset = sortrows(rawalgset);
    toDelete = false(nAlgs, 1);
    for i=2:nAlgs
        if all(rawalgset(i, :) == rawalgset(i-1, :))
            toDelete(i) = true;
        end
    end
    
    rawalgset(toDelete, :) = [];
end