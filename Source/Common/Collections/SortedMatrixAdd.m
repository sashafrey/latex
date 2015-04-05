function [ matrix, id ] = SortedMatrixAdd(matrix, key, value)
    if (isnan(matrix.KeyL))
        matrix.KeyL = size(key, 2);
        if (nargin < 3)
            matrix.ValueL = 0;
        else
            matrix.ValueL = size(value, 2);
        end
    end

    for i = 1:size(key, 1)
        [contains, index] = SortedMatrixContains(matrix, key(i, :));

        if (contains)
            % don't replace the value - this is by design.
            continue;
        end

        if (nargin < 3)
            matrix = VectorAdd(matrix, key(i, :));
        else            
            matrix = VectorAdd(matrix, [key(i, :), value(i, :)]);
        end
        
        id = matrix.Count;
        matrix.Idx = VectorAdd(matrix.Idx, id);
        matrix.Idx.Data((index + 1) : id) = matrix.Idx.Data(index : (id - 1));
        matrix.Idx.Data(index) = uint64(id);
    end
end
