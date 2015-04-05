function [ vector ] = CellVectorAdd( vector, value )
    % Adds a value into cell vector.
    curCapacity = length(vector.Data);
    valuesCount = 1;
    
    if (valuesCount + vector.Count > curCapacity)
        lenInc = max([valuesCount, curCapacity]);
        vector.Data = [vector.Data; cell(lenInc, 1)];
    end
    
    vector.Data{vector.Count + 1} = value;
    vector.Count = vector.Count + valuesCount;
end

