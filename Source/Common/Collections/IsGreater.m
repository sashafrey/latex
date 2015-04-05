function isGreater = IsGreater(vector1, vector2)
    if (~IsGeq(vector1, vector2))
        isGreater = false;
        return;
    end

    isGreater = any(vector1 ~= vector2);
end