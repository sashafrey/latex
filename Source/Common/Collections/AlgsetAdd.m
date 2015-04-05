function [ algset, id ] = AlgsetAdd(algset, algorithms)
    if (isnan(algset.L))
        algset.L = size(algorithms, 2);
    end

    algorithms = PackLogicals(algorithms);
    [algset, id] = SortedMatrixAdd(algset, algorithms);
end

