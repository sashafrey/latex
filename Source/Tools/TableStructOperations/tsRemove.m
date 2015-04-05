function TableStructFiltered = DelFromTableStruct(S,delidx)
%Функция удаляет из структуры подмножество значений из каждого поля
%Предполагается, что структура имеет формат вида (формат "таблица"):
% S.field1[Nx1 | NxM1]
% S.field2[Nx1 | NxM2]
% ...
% S.fieldk[Nx1 | NxMk]
% [Daniel, 22.12] Если длина поля не соответствует длине индексов, поле
% игнорируется

Check(tsIsValid(S));

SFields = fieldnames(S);
TableStructFiltered = S;
if islogical(delidx), delidx = find(delidx); end
maxidx = max(delidx);
if isempty(delidx), return; end
for f=1:length(SFields)
    if maxidx > length(TableStructFiltered.(SFields{f})), continue; end
    if size(TableStructFiltered.(SFields{f}),1) == length(delidx)
        TableStructFiltered.(SFields{f}) = [];
    elseif size(TableStructFiltered.(SFields{f}),1) > 1 && size(TableStructFiltered.(SFields{f}),2) > 1
        TableStructFiltered.(SFields{f})(delidx,:) = [];
    else
        TableStructFiltered.(SFields{f})(delidx) = [];
    end
end

if (ismember('IndexCache', fieldnames(S)))
    TableStructFiltered.IndexCache = [];
end

Check(tsIsValid(TableStructFiltered));

end