function grade = getSampleGrade_PacBayes(X, Y, classifierLearner, boundType)
    
    w = classifierLearner(X, Y);
    
    X = [X, ones(size(X, 1), 1)];

    margins = Y .* (X * w) ./ ...
        sqrt(sum(X .^ 2, 2));

    if strcmp(boundType, 'dd')
        curr_d = size(X, 2);
        grade = DDmargin(margins, curr_d);
    elseif strcmp(boundType, 'di')
        grade = DImargin(margins);
    else
        Check(0 == 1);
    end
end