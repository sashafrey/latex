function [algs] = BuildLevel(sampleSize, errorLevel)
    numAlgs = nchoosek(sampleSize, errorLevel);
    ind = nchoosek(1:sampleSize, errorLevel);
    algs = zeros( numAlgs, sampleSize );
    for n = 1:numAlgs
        algs(n, ind(n, :) ) = 1;
    end
end