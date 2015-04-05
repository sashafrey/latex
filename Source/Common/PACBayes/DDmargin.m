function bound = DDmargin(gamma, d, delta)
    % Dimension-dependent PAC Bayes bound from "Dimensionality Dependent 
    % PAC-Bayes Margin Bound".
    %
    % bound = DDMARGIN(gamma, d), where 
    %   - gamma is the vector of margins for all items.  Condition 
    %     "gamma(i) <= 0" is equivalent to "classifier makes an error 
    %     on x(i)".
    %   - d is the dimension of the feature space.
    %   - delta defines the desired level of confidence in the bound 
    %     (bound holds with probability of 1-delta.
    %     default = 0.1
    %
    % Example:
    %  bound = DDmargin(1.0 - 0.9*(rand(1000, 1)), 15);
    
    if (~exist('delta', 'var'))
        delta = 0.1;
    end
    
    if (size(gamma, 2) ~= 1)
        % gamma vector must be oriented as column, and not as row.
        gamma = gamma';
    end

    % Search space for mu 
    mu = exp(-7:0.01:15);  % aprox. [0.001, 3000000]
    
    n = length(gamma);  % number of items
       
    % rhs is the bound for kl(er_s, er_d)
    rhs = (d / 2 * log(1 + mu .* mu / d) + log((n + 1) / delta)) / n;

    % empirical error
    er_s = mean(1 - normcdf(gamma * mu, 0, 1));

    bound = zeros(length(mu), 1);
    for i=1:length(mu)
        % er_d is a bound for stochastic Gibbs classifier
        er_d = inv_kl(er_s(i), rhs(i));

        % multiply er_d by two is the easiest way to obtain simple bound 
        % for non-stochastic classifier.
        bound(i) = 2 * er_d; 
    end

    bound = min(bound);
end