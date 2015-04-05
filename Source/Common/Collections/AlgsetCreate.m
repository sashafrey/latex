function [ algset ] = AlgsetCreate(algset_raw)
    algset = SortedMatrixCreate();
    algset.L = NaN;
    
    if (exist('algset_raw', 'var'))
        algset = AlgsetAdd(algset, algset_raw);
    end
end