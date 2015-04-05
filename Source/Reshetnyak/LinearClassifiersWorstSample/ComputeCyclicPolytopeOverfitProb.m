function [res, dp] = ComputeCyclicPolytopeOverfitProb(dim, sampleSize, trainSize, eps)
    errLevel = ceil(eps * (sampleSize - trainSize));
    dp = zeros(sampleSize + 2, trainSize + 2, dim + 1, errLevel + 1);
    
    dp(1, 1, 1, 1) = 1;
    for L = 1 : sampleSize + 1
        for t = 1 : min(L, trainSize + 1);
            for d = 1 : dim + 1
                for m = 1 : errLevel + 1
                    for next = L + 1 : sampleSize + 2
                        dp(next, t + 1, d, m) = dp(next, t + 1, d, m) + dp(L, t, d, m);
                        if d < dim && next > L + 1 && m + next - L <= errLevel + 2
                            if t == 1 || next == sampleSize + 2
                                dp(next, t + 1, d + 1, m + next - L - 1) = dp(next, t + 1, d + 1, m + next - L - 1) + dp(L, t, d, m);
                            else
                                dp(next, t + 1, d + 2, m + next - L - 1) = dp(next, t + 1, d + 2, m + next - L - 1) + dp(L, t, d, m);
                            end
                        end
                    end
                end
            end
        end
    end
    
  
    res = sum(dp(sampleSize + 2, trainSize + 2, :, errLevel + 1))
end