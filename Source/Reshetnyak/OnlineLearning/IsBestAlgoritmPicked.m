function [res] = IsBestAlgoritmPicked(totalError, trainError, nextObject)
    res = max(totalError(trainError == min(trainError))) == min(totalError);
end