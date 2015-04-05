function [answers] = SVMClassify(weights, trainSample, Kernel, sample)

    [n dim] = size(sample);
    answers = zeros(n, 1);
    score = -weights(size(trainSample, 1) + 1) * ones(n, 1);
    
    for i = 1:size(trainSample,1)
        if (weights(i) ~= 0)
             for j = 1:n
                score(j) = score(j) + weights(i) * Kernel( trainSample(i, :), sample(j, :) );
             end
        end
    end
    
    answers = sign(score);
    answers(answers == 0) = 1;
   
end