sampleSize = 21
errorLevel = 5;
        
trainSize = floor(sampleSize / 2);
eps = 0.01;
ballCenterError = 4;
radius = 4;
changeCount = floor((radius + ballCenterError - errorLevel)/2);
maxTrainError = TrainErrorOverfitThreshold(sampleSize, trainSize, errorLevel, eps);
nchoosek(sampleSize, trainSize)
if maxTrainError >= 0
    myEstimation = HypergeometricTail(changeCount + maxTrainError, sampleSize, trainSize, ballCenterError)
else
    myEstimation = 0
end
%otherEstimation = ComputeOtherBallLayerIntersectionBound(sampleSize, trainSize, errorLevel, radius, eps)

layer = nchoosek([1:sampleSize], errorLevel);
algs = false(1, sampleSize);
pos = 0;
center = [true(1, ballCenterError) false(1, sampleSize - ballCenterError)];
for n = 1:size(layer, 1)
    newAlg = false(1, sampleSize);
    newAlg(layer(n, :)) = true;
    if sum(center ~= newAlg) <= radius
        pos = pos + 1;
        algs(pos, :) = newAlg;
    end 
end
if pos == size(algs, 1)
    exactValue = nchoosek(sampleSize, trainSize) * ExactFunctional(algs, trainSize, eps)
end

%     if ( abs(myEstimation - exactValue) > 0.5 || abs(myEstimation - otherEstimation) > 0.5)
%         sampleSize
%         errorLevel
%         nchoosek(sampleSize, trainSize)
%         myEstimation
%         otherEstimation
%         exactValue
%     end

