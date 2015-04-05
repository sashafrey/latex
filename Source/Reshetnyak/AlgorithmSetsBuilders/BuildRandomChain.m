function [algs] = BuildRandomChain(sampleSize, minLevel, numAlgs, GetNext)
    
    algs = false(numAlgs, sampleSize);
    algs(1, 1 : minLevel) = true;
    for n = 2:numAlgs
        noNewAlgorithm = true;
        while noNewAlgorithm
            algs(n, :) = GetNext(algs(n - 1, :), n);
            noNewAlgorithm = false;
            for i = 1 : n - 1
                if all(algs(n, :) == algs(i, :))
                    noNewAlgorithm = true;
                end
            end
        end
    end
   
end