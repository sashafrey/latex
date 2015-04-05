function TableLength = tsLength(tableStruct)
    % Check(tsIsValid(tableStruct));
    % функция проверяет, является ли структура tableStruct 
    % пустой (нулевой длины).

    if (isempty(tableStruct))
        TableLength = 0;
        return;
    end
    
    fields = fieldnames(tableStruct);
    
    % Удаляем из списка полей IndexCache.
    for i=1:length(fields)
        if (strcmp(fields{i}, 'IndexCache'))
            fields(i) = [];
            break;
        end
    end

    if (isempty(fields))
        TableLength = 0;
    else 
        TableLength = size(tableStruct.(fields{1}), 1);
    end 
end