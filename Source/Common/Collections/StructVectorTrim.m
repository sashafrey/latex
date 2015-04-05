function [ vector ] = StructVectorTrim( vector )
    vector.Data = vector.Data(1:vector.Count, :);
end
