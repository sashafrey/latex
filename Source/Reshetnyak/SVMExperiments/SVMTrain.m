function [classificationRule] = SVMTrain(sample, answers, Kernel, penalty)

    [sampleSize dim] = size(sample);    
    scalarProductMatrix = zeros(sampleSize);
    
    for i = 1:sampleSize
        for j = 1:sampleSize
            scalarProductMatrix(i, j) = answers(i) * answers(j) * Kernel(sample(i, :), sample(j, :) );
        end
    end
    
    [lambda,funcValue,exitflag,output]= quadprog(scalarProductMatrix, -ones(sampleSize, 1), ...
        zeros(1, sampleSize), 0, answers',0, ...
        zeros(sampleSize, 1), penalty * ones(sampleSize, 1) );
    weights = (lambda .* answers)' * sample;
    bias = median( sample(lambda > 0, :) * weights - answers(lambda > 0) );
    classificationRule = [(lambda .* answers)' bias];
end