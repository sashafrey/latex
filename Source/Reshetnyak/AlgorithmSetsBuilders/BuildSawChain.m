function [algs] = BuildSawChain(sampleSize, minLevel, numAlgs)

    function [alg] = GetNext(alg, algNumber)
        changeTo = (rem(algNumber, 2) == 0);
        changePos = ceil(numel(alg) * rand(1));
        while alg(changePos) == changeTo
            changePos = ceil(numel(alg) * rand(1));
        end
        alg(changePos) = ~alg(changePos);
    end
    
    algs = BuildRandomChain(sampleSize, minLevel, numAlgs, @GetNext);
end