sampleSize = 50
errorLevel = 8;
centerError = 8;       
trainSize = floor(sampleSize / 2);
eps = 0.001;
nchoosek(sampleSize, trainSize)
maxRadius = centerError + errorLevel;
chooseTable = ComputeChooseTable(sampleSize);

ballEstimation = zeros(1, maxRadius);
ballNumAlgs = zeros(1, maxRadius);
for radius = 1:maxRadius
    changeCount = floor((radius + centerError - errorLevel)/2);
    %maxTrainError = 0; %TrainErrorOverfitThreshold(sampleSize, trainSize, errorLevel, eps);
    lb = max(centerError - errorLevel, 0);
    ballEstimation(radius) = HypergeometricTail(sampleSize, trainSize, centerError, changeCount) / ...
       chooseTable(sampleSize + 1, trainSize + 1);
    ballNumAlgs(radius) = HypergeometricSum(sampleSize, errorLevel, centerError, centerError - changeCount, centerError - lb);
end

lexFirstSetEstimation = zeros(1, maxRadius);
for radius = 1:maxRadius
    lexFirstSetEstimation(radius) =  ComputeOverfitProbabiltyForLexicographicallyFirstSet ...
        ( ballNumAlgs(radius), sampleSize, trainSize, errorLevel, chooseTable);
end

ballNumAlgs
ballEstimation
lexFirstSetEstimation
if any(lexFirstSetEstimation > ballEstimation)
    'Ball Better'
end
hold on
plot(ballNumAlgs, ballEstimation, 'r');
plot(ballNumAlgs, lexFirstSetEstimation, 'b');
