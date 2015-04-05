function [ vector ] = VectorCreate( values )
    % Array with incremental memory allocation.
    % Works like std::vector ? ?++, add List<T> in C#.
    % The analog of std::vector.operator[int i] would be
    % vector.Data(i, :). This means that the structure supports up to 2
    % dimentions, and uses first dimentions for indexing.
    % Use VectorCreate, VectorAdd, VectorTrim to work with this data
    % structure.
    
    vector.Data = [];
    vector.Count = 0;
    
    if (exist('values', 'var'))
        vector = VectorAdd(vector, values);
    end
end

