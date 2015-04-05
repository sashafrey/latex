function DynArray = CreateDynArray(Values)
    % Создает динамический массив, допускающий экспоненциальное
    % нарасщивание используемой памяти.
    % Для работы с DynArray используйте операции CreateDynArray, 
    % PushToDynArray, DynArray2Array.
    
    DynArray.Array = [];
    DynArray.RecordsCount = 0;
    
    if (exist('Values', 'var'))
        DynArray = PushToDynArray(Values);
    end
end