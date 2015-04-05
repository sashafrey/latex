function [objectsDistr] = GetBoundedDistanceDistribution(maxDist, numObjects, time)
    if nargin < 3
        time = numObjects;
    end
    objectsDistr = zeros(time, numObjects);
    for t = 1 : time
        admissableObjects = max(1, t - maxDist) : min(numObjects, t + maxDist);
        objectsDistr(t, admissableObjects) = 1 / numel(admissableObjects);
    end
    objectsDistr;
end