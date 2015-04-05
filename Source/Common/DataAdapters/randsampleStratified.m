function train = randsampleStratified(target, ell)
    targetLabels = unique(target);
    Check(length(targetLabels) == 2);
    
    nItems = length(target);
    if (ell > nItems)
        ell = nItems;
    end
    
    class1ids = find(target == targetLabels(1));
    class2ids = find(target == targetLabels(2));
    
    class1count = floor(length(class1ids) * ell / nItems);
    class2count = ell - class1count;
    
    trainClass1 = class1ids(randsample(length(class1ids), class1count));
    trainClass2 = class2ids(randsample(length(class2ids), class2count));
    train = false(1, length(target));
    train(trainClass1) = true;
    train(trainClass2) = true;    
end

