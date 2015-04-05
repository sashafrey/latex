function foldIds = GenerateNFoldCVIndexes(target, n)
    classes = unique(target);
    
    foldIds = NaN(length(target), 1);
    for iFold = n : -1 : 1
        for iTarget = classes'
            curTarget = find(target == iTarget);
            nItems1 = length(curTarget);
            nItemsFold1 = floor(nItems1 / iFold);
            ids = curTarget(randsample(nItems1, nItemsFold1));
            foldIds(ids) = iFold;
            target(ids) = NaN;            
        end        
    end
end
