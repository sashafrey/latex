function [bound, optClassification] = TransductiveExactERM(algs, trainSample, trainSampleClasses, eps)
    [numAlgs sampleSize] = size(algs);
    controlSize = sampleSize - numel(trainSample);
    numAlgs
    trainSampleClasses(trainSampleClasses == -1) = 0;
    bound = [inf; 0];
    optClassification= false(2, sampleSize);
   
    for numOnes = 0:controlSize
        %possClasses = nchoosek(setdiff(1:sampleSize,trainSample),numOnes);
        possClasses = nchoosek(1:sampleSize,numOnes);
        for n = 1:size(possClasses, 1)
            curClassification = false(1, sampleSize);
            %curClassification(trainSample) = (trainSampleClasses == 1);
            curClassification(possClasses(n, :)) = true;
            curBound = ExactFunctional(algs ~= repmat(curClassification, size(algs,1), 1), ...
                                       sampleSize - controlSize, eps);
            if bound(2) < curBound
                optClassification(2, :) = curClassification;
                bound(2) = curBound 
            end
            if bound(1) > curBound
                optClassification(1, :) = curClassification;
                bound(1) = curBound
            end
            
        end
    end
     
end