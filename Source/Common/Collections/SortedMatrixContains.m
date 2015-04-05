function [contains, index] = SortedMatrixContains(matrix, line)
    low = 1;
    high = matrix.Count;
    
    while (low <= high)
        mid = round((low + high)/2);
        curLine = matrix.Data(matrix.Idx.Data(mid), 1:matrix.KeyL);
        
        if (IsLexLess(line, curLine))
            high = mid - 1;
        elseif (IsLexGreater(line, curLine))
            low = mid + 1;
        else
            contains = true;
            index = mid;
            return;
        end
    end
    
    contains = false;
    index = low;
end
