function [ key_value ] = SortedMatrixGet(matrix, index)
    key_value = matrix.Data(matrix.Idx.Data(index), :);
end

