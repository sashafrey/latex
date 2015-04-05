function [ vector ] = StructVectorAdd( vector, values )
    % Adds values into vector.
    
    if (isempty(values))
        return;
    end
    
    if size(values, 1) > 1 && size(values, 2) > 1
        error('Values passed to StructVectorAdd should be one-dimensional!');
    end
    
    if size(values, 1) > 1
        values = values';
    end
    
    curCapacity = size(vector.Data, 1);
    valuesCount = size(values, 1);
    
    if isempty(vector.Data)
        curCapacity = 0;
    end
    
    if (valuesCount + vector.Count > curCapacity)
        lenInc = max([valuesCount, curCapacity]);
        vector.Data = [vector.Data; ...
            repmat(values(1), lenInc, 1)];
    end
    
    addFrom = vector.Count + 1;
    addTo = vector.Count + valuesCount;
    vector.Data(addFrom:addTo, :) =  values;
    
    vector.Count = vector.Count + valuesCount;
end

