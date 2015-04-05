function [contains, index] = AlgsetContains(algset, algorithm)
    if (islogical(algorithm))
        algorithm = PackLogicals(algorithm);
    end
    
    [contains, index] = SortedMatrixContains(algset, algorithm);
end