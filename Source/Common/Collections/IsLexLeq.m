function isLexLeq = IsLexLeq(vector1, vector2)
    for i = 1:length(vector1)
        if (vector1(i) < vector2(i))
           isLexLeq = true;
           return;
        elseif (vector1(i) > vector2(i))
            isLexLeq = false;
            return;
        end 
    end
    
    isLexLeq = true;    
end