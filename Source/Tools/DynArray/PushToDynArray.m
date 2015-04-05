function DynArray = PushToDynArray(DynArray, Values)
    % �������� ������ � ������������ ������.
    % Values --- �������� ������-������ (�� cell-array).
    % ���� ������ DynArray �� ������, �� ������ ������ ����� �� �� "������",
    % ��� � ��� ����������. ����� ������� error.
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
