function Array = DynArray2Array(DynArray)
    Array  = DynArray.Array(1:DynArray.RecordsCount, :);
end