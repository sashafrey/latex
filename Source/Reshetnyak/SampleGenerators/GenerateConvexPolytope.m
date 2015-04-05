function [sample, sampleClasses] = GenerateConvexPolytope(sampleSize, dim)
    sample = zeros(sampleSize, dim);
    for n = 1 : sampleSize
        sample(n, :) = (n / sampleSize).^[1 : dim];
    end
    sampleClasses = zeros(sampleSize, 1);
end