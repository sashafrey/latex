function [res] = MonotoneChainCCV(T, m, length)
    ct = ComputeChooseTable(T);
    res = zeros(1, T - 1);
    for t = 1 : T - 1
        for d = 0 : min(length, T - t)
            cf = d + m * (T - d - t) / (T - d - 1);
            if d == length
                cf = d * (T - d) / t +  m * (T - d - t) / t;
            end
            res(t) = res(t) + cf * ct(T - d, t);
        end
        res(t) = res(t) / (T - t) / ct(T + 1, t + 1);
    end
end