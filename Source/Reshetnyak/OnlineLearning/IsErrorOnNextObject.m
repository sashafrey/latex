function [res] = IsErrorOnNextObject(totalError, trainError, nextObject)
    res = (max(nextObject(trainError == min(trainError))) == true);
end