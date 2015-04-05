function bound = DImargin(gamma, delta)
    % Dimention-independent PAC Bayes bound from "Dimensionality Dependent 
    % PAC-Bayes Margin Bound".
    %
    % bound = DIMARGIN(gamma), where 
    %   - gamma is the vector of margins for all items. 
    %     Condition "gamma(i) <= 0" is equal to "classifier makes an error on x(i)".
    %   - delta defines the desired level of confidence in the bound 
    %     (bound holds with probability of 1-delta.
    %     default = 0.1
    %
    % Example:
    %  bound = DImargin(1.0 - 0.9*(rand(1, 1000)));
    
    if (~exist('delta', 'var'))
        delta = 0.1;
    end
    
    f = @(mu)DImargin_mu(gamma, delta, mu);
    x = exp(-7:0.01:15);
    y = f(x);
    bound = min(y);    
end

function bound = DImargin_mu(gamma, delta, mu_vec)
    % This bound must be optimized across all values of mu > 0.
    bound = zeros(length(mu_vec), 1);

    n = length(gamma);  % number of items
    for i=1:length(mu_vec)
        mu = mu_vec(i);

        % rhs is the bound for kl(er_s, er_d)
        rhs = (mu * mu / 2 + log((n + 1) / delta)) / n;

        % empirical error
        er_s = mean(1 - normcdf(mu * gamma, 0, 1));

        % er_d is a bound for stochastic Gibbs classifier
        er_d = inv_kl(er_s, rhs);

        % multiply er_d by two is the easiest way to obtain simple bound for
        % non-stochastic classifier.
        bound(i) = 2 * er_d; 
    end
end