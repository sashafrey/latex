function DynArray = PushToDynArray(DynArray, Values)
    % ѕомещает данные в динамический массив.
    % Values --- числова€ вектор-строка (не cell-array).
    % ≈сли массив DynArray не пустой, то данные должны иметь ту же "ширину",
    % что и все предыдущие. »наче вылетит error.
    if (isempty(Values))
        return;
    end
    
    curCapacity = size(DynArray.Array, 1);
    valuesCount = size(Values, 1);
    
    if (~isempty(DynArray.Array))
        if (size(DynArray.Array, 2) ~=  size(Values, 2))
            error('unable to put values to dyn-array due to unsiutable length(Values).');
        end
    end
    
    if (valuesCount + DynArray.RecordsCount > curCapacity)
        lenInc = max([valuesCount, curCapacity]);
        DynArray.Array = [DynArray.Array; NaN(lenInc, size(Values, 2))];
    end
    DynArray.Array(DynArray.RecordsCount + 1 : DynArray.RecordsCount + valuesCount, :) =  Values;
    DynArray.RecordsCount = DynArray.RecordsCount + valuesCount;
end
