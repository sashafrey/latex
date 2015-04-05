function [algs] = BuildMonotoneChain(sampleSize, minLevel, numAlgs)
    algs = zeros(numAlgs, sampleSize);
    algs(:, 1 : minLevel) = 1;
    algs(1 : numAlgs - 1, minLevel + 1 : minLevel + numAlgs - 1) = hankel(ones(numAlgs - 1, 1));
end