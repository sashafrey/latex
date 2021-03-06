function [threshold] = TrainErrorOverfitThreshold(sampleSize, trainSize, numErrors, eps)
    threshold = floor( trainSize * (repmat(numErrors, size(eps)) - ... 
                repmat(eps * (sampleSize - trainSize), size(numErrors) ) ) / sampleSize + ... 
                repmat(eps, size(numErrors)) / 3 / sampleSize);
end