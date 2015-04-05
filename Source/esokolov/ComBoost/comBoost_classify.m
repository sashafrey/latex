function [Y, margins] = comBoost_classify(X, ensembleWeights, Y)
    
    if exist('Y', 'var')
        computeMargins = true;
    else
        computeMargins = false;
    end

    L = size(X, 1);
    
    X = [X, ones(L, 1)];
    
    votes = zeros(L, 1);
    margins = zeros(L, 1);
    
    for i = 1:size(ensembleWeights, 1)
        if sum(isnan(ensembleWeights(i, :))) ~= 0
            continue;
        end
        
        dot_products = th_func(X * ensembleWeights(i, :)');
        
        votes = votes + dot_products;
        if computeMargins
            margins = margins + dot_products .* Y;
        end
    end
    
    Y = sign(votes);
end

function y = th_func(x)
    y = 2 * (1 ./ (1 + exp(-x))) - 1;
end
