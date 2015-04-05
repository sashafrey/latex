function StructSorted = tsSort(S,FieldName)
%‘ункци€ упор€дочивает содержимое структуры по заданному полю
%ѕредполагаетс€, что структура имеет формат вида (формат "таблица"):
% S.field1[Nx1 | NxM1]
% S.field2[Nx1 | NxM2]
% ...
% S.fieldk[Nx1 | NxMk]
% ѕри этом поле, по которому мы сортируем, об€зательно должно быть Nx1.

Check(tsIsValid(S));

if ~isfield(S,FieldName), 
    warning('Cash4Cast:Utils:SortTableStruct',['Could not find the field ' FieldName 'in the structure!']);
    StructSorted = S;
    return;
end;

SFields = fieldnames(S);
ToSort = getfield(S,FieldName);
try
    [FieldSorted sortidx] = sort(ToSort);
catch exp
    StructSorted = S;
    warning('Cash4Cast:Utils:SortTableStruct',['Could not sort the contents of the field ' FieldName '!']);
    return;
end
StructSorted = [];
numFields = length(SFields);
for f = 1:numFields
    %if length(S.(SFields{f})) <= 1, continue; end
    if strcmp(SFields{f},FieldName), 
        StructSorted = setfield(StructSorted,FieldName,FieldSorted);
        continue;
    end
    Content = getfield(S,SFields{f});
    StructSorted = setfield(StructSorted,SFields{f},Content(sortidx,:));
end

Check(tsIsValid(StructSorted));

end