function [ matrix ] = SortedMatrixCreate()
    matrix = VectorCreate();
    
    % Ids is the permutation which sorts the matrix.
    % E.g. matrix.Data(Idx(1), :) <= matrix.Data(Idx(2), :) <= ...
    matrix.Idx = VectorCreate();    
    matrix.KeyL = NaN;
    matrix.ValueL = NaN;
end