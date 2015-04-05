function isLexLess = IsLexLess(vector1, vector2)
    for i = 1:length(vector1)
        if (vector1(i) < vector2(i))
           isLexLess = true;
           return;
        elseif (vector1(i) > vector2(i))
            isLexLess = false;
            return;
        end 
    end
    
    isLexLess = false;    
end