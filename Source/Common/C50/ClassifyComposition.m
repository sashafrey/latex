function [predictedClasses, margin] = ClassifyComposition(composition, terms, task, confidence)
    % confidence = minimal margin to make decision. For margins less than confidence use default class.
    if (exist('confidence', 'var'))
        confidence = 0;     
    end
    
    coverage = CalcRulesCoverage(composition, terms, task);
    
    if (isfield(composition,'weight'))
        nItems = task.nItems;
        w1 = composition.weight(composition.target == 1);
        w2 = composition.weight(composition.target == 2);
        t1 = zeros(nItems, 1);
        t2 = zeros(nItems, 1);
        
        for i = 1:nItems
            t1(i) = sum(w1(coverage(composition.target == 1, i)));
            t2(i) = sum(w2(coverage(composition.target == 2, i)));        
        end
    else    
        t1 = sum(coverage(composition.target == 1, :), 1);
        t2 = sum(coverage(composition.target == 2, :), 1);
    end
    
    t = t1 - t2;
    target1 =    t  >= confidence;
    target2 = ((-t) >= confidence);
    predictedClasses = NaN(task.nItems, 1);
    predictedClasses(target1) = 1;
    predictedClasses(target2) = 2; 
    margin = zeros(task.nItems, 1);
    margin(task.target == 1) =  t(task.target == 1);
    margin(task.target == 2) = -t(task.target == 2);
    
    tMax = composition.tailClass(tsLength(composition));
    predictedClasses(isnan(predictedClasses)) = tMax;
end