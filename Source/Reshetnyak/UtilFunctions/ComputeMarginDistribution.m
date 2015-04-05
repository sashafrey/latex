function [margins] = ComputeMarginDistribution(pickedAlg, algs)
    [numAlgs numObjects] = size(algs);
    margins = inf(1, numObjects); 
    
    for n = 1 : numAlgs
        d = sum( xor(pickedAlg, algs(n, :)) & algs(n, :) );
        for i = 1 : numObjects
            if pickedAlg(i) ~= algs(n, i)
                margins(i) = min(margins(i), d);
            end
        end
    end
        
    signs = ones(1, numObjects);
    signs(pickedAlg == true) = -1;
    margins = margins .* signs;
end
