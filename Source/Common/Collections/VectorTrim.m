function [ vector ] = VectorTrim( vector )
    vector.Data = vector.Data(1:vector.Count, :);
end