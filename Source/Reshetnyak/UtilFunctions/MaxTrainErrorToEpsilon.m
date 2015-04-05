function [eps] = MaxTrainErrorToEpsilon(sampleSize, trainSize, errorLevel, maxTrainError)
    eps = (errorLevel - maxTrainError) / (sampleSize - trainSize) - maxTrainError / trainSize - 0.1 / sampleSize;
end