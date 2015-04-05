function [overfitProb] = TwoAlgorithmsBound(algs, objectsDistr)
    [maxTime numObjects] = size(objectsDistr)
    if numObjects ~= size(algs, 2)
        'ERROR!!! algs and objectDistr sizes does not match'
        return
    end
    overfitProb = zeros(1, maxTime - 1);
    dp = zeros(sum(algs(1, :)) + 1, sum(algs(2, :)) + 1, maxTime);
    dp(1, 1, 1) = 1;
    for t = 2 : maxTime
        fisrtError = objectsDistr(t, :) * algs(1, :)';
        secondError = objectsDistr(t, :) * algs(2, :)';
        for s1 = 1 : size(dp, 1)
            for s2 = 1 : size(dp, 2)
                for n = 1 : numObjects
                    if s1 > algs(1, n) && s2 > algs(2, n)
                        dp(s1, s2, t) = dp(s1, s2, t) + objectsDistr(t - 1, n) * dp(s1 - algs(1, n), s2 - algs(2, n), t - 1);
                    end
                 end
                if s1 <= s2
                    overfitProb(t - 1) = overfitProb(t - 1) + fisrtError * dp(s1, s2, t) ;
                else
                    overfitProb(t - 1) = overfitProb(t - 1) + secondError * dp(s1, s2, t) ;
                end
            end
        end
    end
end