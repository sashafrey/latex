function c = cnk(n, k)
    if (n >= 0 && k >= 0 && k <= n)
        %c = nchoosek(n, k);
        c = exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1));
    else
        c = 0;
    end
end