function [vector] = StructVectorDel(vector, delIdx)
    if delIdx > vector.Count
        return;
    end
    
    vector.Count = vector.Count - 1;
    vector.Data(delIdx) = [];
end