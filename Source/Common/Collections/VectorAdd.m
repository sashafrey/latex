function [ vector ] = VectorAdd( vector, values )
    % Adds values into vector.
    % If vector is not empty then size(values, 2) must be equal to
    % size(vector, 2).
    
    if (isempty(values))
        return;
    end
    
    curCapacity = size(vector.Data, 1);
    valuesCount = size(values, 1);
    
    if (~isempty(vector.Data))
        if (size(vector.Data, 2) ~=  size(values, 2))
            error('unable to put values to dyn-array due to unsiutable length(values).');
        end
    end
    
    if (valuesCount + vector.Count > curCapacity)
        lenInc = max([valuesCount, curCapacity]);
        if (isinteger(values))
            vector.Data = [vector.Data; repmat(uint64(0), lenInc, size(values, 2))];
        else
            vector.Data = [vector.Data; zeros(lenInc, size(values, 2))];
        end
    end
    
    if (valuesCount == 1)
        vector.Data(vector.Count + 1, :) =  values;
    else
        vector.Data(vector.Count + 1 : vector.Count + valuesCount, :) =  values;
    end
    
    vector.Count = vector.Count + valuesCount;
end

