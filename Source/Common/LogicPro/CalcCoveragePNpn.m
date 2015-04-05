function [P, N, p, n] = CalcCoveragePNpn(coverage, correct, rules, params, weights)
    if (~exist('weights', 'var'))
        p = sum(coverage & correct, 2);
        n = sum(coverage & ~correct, 2);
        P = sum(correct, 2);
        N = sum(~correct, 2);
    else
        nRules = size(coverage, 1);
        p = zeros(nRules, 1);
        n = zeros(nRules, 1);
        P = zeros(nRules, 1);
        N = zeros(nRules, 1);
        
        for i=1:nRules
            p(i) = sum(weights(coverage(i,:) & correct(i, :)));
            n(i) = sum(weights(coverage(i,:) & ~correct(i, :)));
            P(i) = sum(weights(correct(i, :)));
            N(i) = sum(weights(~correct(i, :)));
        end
    end
    
    [P, N, p, n] = params.fAdjust(P, N, p, n, rules);
end