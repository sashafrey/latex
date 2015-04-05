function [ sau_storage ] = CreateSAU(max_dim)
    sau_storage = SortedMatrixCreate();
    sau_storage = SortedMatrixAdd(sau_storage, [0, zeros(1, max_dim)], 1);
end
