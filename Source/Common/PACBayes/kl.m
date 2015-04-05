function kl_value = kl(p, q)
    % KL(p, q) --- the Kullback-Leibler divergence for Bernoulli random
    % variables with probabilities p and q.
    %
    % k = KL(p, q) supports the following:
    %   - 'p' and 'q' are scalar values
    %   - 'p' and 'q' are arrays of the same length
    %   - either 'p' or 'q' is scalar, and other value is a vector.
    
    kl_value = p .* log(p ./ q) + (1 - p) .* log( (1 - p) ./ (1 - q));
end