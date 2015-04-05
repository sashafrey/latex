function isLexGreater = IsLexGreater(vector1, vector2)
    for i = 1:length(vector1)
        if (vector1(i) > vector2(i))
           isLexGreater = true;
           return;
        elseif (vector1(i) < vector2(i))
            isLexGreater = false;
            return;
        end 
    end
    
    isLexGreater = false;    
end