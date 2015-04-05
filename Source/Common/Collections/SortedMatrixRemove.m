function [ matrix ] = SortedMatrixRemove(matrix, key)
    for i = 1:size(key, 1)
        [contains, index] = SortedMatrixContains(matrix, key(i, :));

        if (~contains)
            return;
        end

        matrix.Data(index, :) = [];
        matrix.Count = matrix.Count - 1;
        matrix.Idx.Data(matrix.Idx.Data == index) = [];
        matrix.Idx.Data(matrix.Idx.Data > index) = matrix.Idx.Data(matrix.Idx.Data > index) - 1;
    end
end

