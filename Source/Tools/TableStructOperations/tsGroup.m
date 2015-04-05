function GroupedTableStruct = tsGroup(TableStruct, fieldname, GroupedFieldName)
    if (isempty(TableStruct))
        GroupedTableStruct = [];
        return;
    end
    
    if (~exist('GroupedFieldName', 'var'))
        Check(~strcmp('Info', fieldname));
        GroupedFieldName = 'Info';
    end
    Check(tsIsValid(TableStruct));
    Check(isfield(TableStruct, fieldname));
    TableStruct = tsSort(TableStruct, fieldname);
    fieldValues = TableStruct.(fieldname);
    uniqueValues = unique(fieldValues);
    TableStructArray = cell(length(uniqueValues), 1);
    CountArray = zeros(length(uniqueValues), 1);

    j = 1;
    prev = 1;
    for i=1:tsLength(TableStruct)
        if (i~=1) && (fieldValues(i) ~= fieldValues(i-1)) 
            tmp = tsSelect(TableStruct, prev:(i - 1));
            TableStructArray{j} = rmfield(tmp, fieldname);
            CountArray(j) = i - prev;
            prev = i;
            j = j+1;
        end
    end
    TableStructArray{j} = tsSelect(TableStruct, prev:i);
    CountArray(j) = i - prev + 1;
    
    GroupedTableStruct.(fieldname) = uniqueValues;
    GroupedTableStruct.(GroupedFieldName) = TableStructArray;
    GroupedTableStruct.Count = CountArray;
    Check(tsIsValid(GroupedTableStruct));    
end