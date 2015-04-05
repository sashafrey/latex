function isGeq = IsGeq(vector1, vector2)
    len = length(vector1);
    persistent u;
    if (isempty(u) || length(u) ~= len)
        u = repmat(bitcmp(uint64(0)), 1, len);
    end
    isGeq = all(bitor(bitand(vector1, u), bitxor(vector2, u)) == u);
end