function success = tsIsValid(tableStruct)
    % Checks if structure is a valid TableStruct object.

    if (isempty(tableStruct))
        success = 1;
        return;
    end
    if(~isstruct(tableStruct))
        success = 0;
        return;
    end

    success = 1;

    fields = fieldnames(tableStruct);
    fieldsCount = length(fields);
    
    % Remove IndexCache from the list of fields.
    for i=1:fieldsCount
        if (strcmp(fields{i}, 'IndexCache'))
            fields(i) = [];
            fieldsCount = length(fields);
            break;
        end
    end
    
    xSizes = zeros(1, fieldsCount);
    ySizes = zeros(1, fieldsCount);
    for i=1:fieldsCount
        curSizes = size(tableStruct.(fields{i}));
        xSizes(i) = curSizes(1);
        ySizes(i) = curSizes(2);
    end
    
    if (length(unique(xSizes)) ~= 1),
        success = 0; 
    end;
    
    if (xSizes(1) == 1)
        % Table might be oriented in the wrong direction.
        % If xSizes(1) == 1, than either table have just one element, or it
        % is mis-orientated.
        
        % If all-sizes are the same, and greater than 2, than it is
        % wrong-oriented.
        if (length(unique(ySizes)) == 1) && (ySizes(1) > 1)
            success = 0;
        end
    end
end