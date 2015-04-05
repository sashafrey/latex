function permutedVector = MyRandperm(vector)
    [ignore, ind] = sort(rand(1, numel(vector)));
    permutedVector = vector(ind);
end