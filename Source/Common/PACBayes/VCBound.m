function bound = VCBound(dim, n, delta)
    bound = sqrt(2 * dim * log(exp(1) * n / dim) / n) + sqrt(log (1 / delta) / 2 / n);
end