function [algs] = BuildStratifiedChain(sampleSize, minLevel, numAlgs)

    function [alg] = GetNext(alg, algNumber)
        changePos = ceil(numel(alg) * rand(1));
        alg(changePos) = ~alg(changePos);
    end
    
    algs = BuildRandomChain(sampleSize, minLevel, numAlgs, @GetNext);
end