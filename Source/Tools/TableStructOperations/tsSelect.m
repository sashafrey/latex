function sourcestruct = tsSelect(sourcestruct,indx)
% sourcestruct - структура. Функция вырезает из полей структуры элементы
% под индексами, для которых indx == 1

%Check(tsIsValid(sourcestruct));

if (tsLength(sourcestruct) == 0)
    return;
end

names = fieldnames(sourcestruct);
if(~isempty(names))
    for field = names'
        name = field{1,1};
        if (isempty(sourcestruct.(name)))
            continue;
        end
        if (length(sourcestruct.(name))<1)
            continue;
        end
        sourcestruct.(name) =  sourcestruct.(name)(indx,:);
    end
end

%Check(tsIsValid(sourcestruct));

end