function [res] = TwoAlgorithmsCCV(T, m0, m1, m2)
    ct = ComputeChooseTable(T);
    res = zeros(1, T - 1);
    for t = 1 : T - 1
        for s0 = 0 : m0
            for s1 = 0 : m1
                for s2 = 0 : m2
                    if t >= s0 + s1 + s2
                        res(t) = res(t) + ct(m0 + 1, s0 + 1) * ct(m1 + 1, s1 + 1) * ...
                            ct(m2 + 1, s2 + 1) * ct(T - m0 - m1 - m2 + 1, t - s0 - s1 - s2 + 1) * ...
                            ( (m0 - s0) * (s0 < s1) + (m1 - s1) * (s0 >= s1) + (m2 - s2));
                    end
                end
            end
        end
        res(t) = res(t) / (T - t) / ct(T + 1, t + 1);
    end
    
end
