function string = tsToString(TableStruct, tabSpace)
    % ѕредставл€ет содержимое TableStruct в виде строки.
    % ‘ункци€ требует значительных доработок!
    
    if (nargin == 1)
        tabSpace = 1;
    end;
    spaces = char(ones(1, tabSpace) * ' ');

    if (isempty(TableStruct))
        string = sprintf('%[]\n', spaces);
        return;
    end
    
    string = '';
    Check(isstruct(TableStruct));
    fields = fieldnames(TableStruct);
    for i=1:length(fields)
        field = fields{i};
        string = sprintf('%s%s%s', string, spaces, field);

        value = TableStruct.(field);
        sizes = size(value);
        if (length(sizes(sizes ~= 1)) > 1)
            % многомерный массив (двумерный или выше).
            string = sprintf('%s = array with sizes %s\n',  string, ArrayToString(size(value)));
            continue;
        end
        
        if (isstruct(TableStruct.(field)))
            string = sprintf('%s:\n%s', string, ...
                tsToString(value, tabSpace + 4));
        elseif isa(value, 'function_handle')
            string = sprintf('%s = %s\n', string, func2str(value));
        else
            string = sprintf('%s = %s\n', string, ArrayToString(value));
        end
    end
end