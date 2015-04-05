function ManySourcesCount

    function [res] = Calc(n, d)
        res = 0;
        variants = nchoosek(1 : n - 1, d);
        for n = 1 : nchoosek(n - 1, d)
            good = true;
            for i = 1 : d - 1
                if mod(variants(n, i) + variants(n, i + 1), 2) == 0
                    good = false;
                end
            end
            res = res + good;
        end
    end

    d = 4;
    maxL = 40;
    res = zeros(1, maxL);
    for n = d + 1 : maxL;
        res(n) = Calc(n, d);
    end
    %plot(res);
    loglog(res);
    
end