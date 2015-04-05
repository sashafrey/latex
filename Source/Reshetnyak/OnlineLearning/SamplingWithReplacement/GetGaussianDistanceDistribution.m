function [objectsDistr] = GetGaussianDistanceDistribution(std, numObjects, time)
    if nargin < 3
        time = numObjects;
    end
    objectsDistr = zeros(time, numObjects);
    for t = 1 : time
        objectsDistr(t, :) = pdf('Normal', [1 : numObjects], t, std);
        objectsDistr(t, :) = objectsDistr(t, :) / sum(objectsDistr(t, :));
    end
end