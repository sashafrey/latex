function DDMargin_run(task, degree)
    % DDMARGIN_RUN tunes SVM classifier on whole Wdbc sample and calculates 
    % its Distribution Dependent PAC-Bayes Margin Bound.
    %
    % DDMARGIN_RUN(task), where 'task' is a struct with two fields:
    %   - 'target', vector of target classes, '+1' and '-1'.
    %   - 'objects', feature matrix (rows = items, columns = features),
    %     normalized into [0, 1] range, without missing values (no NaN
    %     values allowed).
    %
    % Requirements:
    %   - LibSVM is compiled, and train function is renamed to
    % 'svmtrain2' to avoid name collision with matlab bioinformatics
    % toolbox.
    
    if (~exist('task', 'var'))
        load('task_mat'); % features are already normalized into [0, 1] range.
    end
    
    if (~exist('degree', 'var'))
        degree = 2;
    end

    % train SVM
    svm = svmtrain2(task.target, task.objects, sprintf('-s 0 -t 1 -d %i -r 0 -g 1 -q', degree));

    % calc polynomial kernel
    phi = power(task.objects * svm.SVs', degree);
    w = svm.sv_coef;

    % append const feature
    phi = [phi, ones(size(phi, 1), 1)];
    w = [w; -svm.rho];

    % normalize phi and w
    phi = bsxfun(@rdivide, phi, sqrt(sum(abs(phi).^2,2)));
    w = w / sqrt(w' * w);

    % calc the margin.
    margin = task.target .* (svm.Label(1) * phi * w);

    errorsVector = task.target ~= svmpredict(task.target, task.objects, svm, '-q');
    assert(all((margin <= 0) == errorsVector));

    bound = DDmargin(margin, svm.totalSV, 0.1);

    plot(sort(margin))
    fprintf('DDmargin bound = %.2f\n', bound);
end

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

function x = inv_kl(p, y, eps)
    % INV_KL is an 'inverse' of KL function. 
    %
    % x = INV_KL(p, y) gives the largest value 'x', such as KL(p, x) <= y.
    % x = INV_KL(p, y, eps) allows to specify the precision. Default:0.001.
    
    if (~exist('eps', 'var'))
        eps = 0.001;        
    end
    
    % ToDo: binary search or Newton method might be more efficient here.
    xx  = 0:eps:1;
    id = find(kl(p, xx) <= y, 1, 'last');
    if (isempty(id))
        x = NaN;
    else
        x = xx(id);
    end
end

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