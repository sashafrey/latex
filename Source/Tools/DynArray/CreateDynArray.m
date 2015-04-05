function DynArray = CreateDynArray(Values)
    % ������� ������������ ������, ����������� ����������������
    % ������������ ������������ ������.
    % ��� ������ � DynArray ����������� �������� CreateDynArray, 
    % PushToDynArray, DynArray2Array.
    
    DynArray.Array = [];
    DynArray.RecordsCount = 0;
    
    if (exist('Values', 'var'))
        DynArray = PushToDynArray(Values);
    end
end