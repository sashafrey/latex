function [ vector ] = CellVectorTrim( vector )
    vector.Data = vector.Data(1:vector.Count, :);
end