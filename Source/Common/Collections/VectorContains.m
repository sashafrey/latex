function contains = VectorContains(vector, value)
    contains = any(vector.Data(1 : vector.Count) == value);
end