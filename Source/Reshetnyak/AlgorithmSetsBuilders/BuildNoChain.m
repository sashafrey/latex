function [algs] = BuildNoChain(algs)
    
    function [alg] = GetNext(alg, algNumber)
        alg(1 :  errors(algNumber)) = 1;
        alg(errors(algNumber) + 1 : numel(alg) ) = 0;
        alg = MyRandperm(alg);
    end
    errors = sum(algs, 2);
    algs = BuildRandomChain(size(algs, 2), errors(1), size(algs, 1), @GetNext);
end