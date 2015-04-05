function evd = CalcEvPairwiseDist(ev, diagVal)
    nAlgs = size(ev, 2);
    evd = zeros(nAlgs, nAlgs);
    for i=1:nAlgs
        for j=i+1:nAlgs
            err = sum(ev(:, i) ~= ev(:, j));
            evd(i, j) = err;
        end
    end
    
    evd = evd + evd' + eye(nAlgs) * diagVal;
end