sampleSize = 200;
%eps = [0.01:0.01:0.25];
eps = 0.1;
connectivity = 6;

ermConnectivityBound = zeros(size(eps));
trainSize = sampleSize / 2
testSize = sampleSize - trainSize;

[sample, sampleClasses] = GenerateSimpleSample(sampleSize);
%[sample, sampleClasses] = GenerateCloseClasses(sampleSize);
%figure
%scatter(sample(:,1), sample(:, 2), 50, sampleClasses, 'filled');
algs = BuildLinearSet(sample, sampleClasses);
graph = BuildFamilyGraph(algs);
%PaintAlgorithmsFamily(algs);
vicinitySizes = cellfun('length', graph);
pmqTable = ComputeExtendedPmqTable(sampleSize, trainSize, eps, max(vicinitySizes));
boundM = zeros(1, sampleSize);
boundV = zeros(1, sampleSize);
optimalQ = zeros(1, sampleSize);
for m = 1 : sampleSize
    possibleQ = [0 : connectivity];
    boundV(m) = pmqTable(m + 1, m + 1, 1);
    for q = max(m + connectivity - sampleSize, 0) : min(connectivity, m) 
        t = pmqTable(m - q + 1, m + 1, connectivity - q + 1);
        if t > boundM(m)
            boundM(m) = t;
            optimalQ(m) = q;
        end
    end
end
bound = ErmConnectivityBound(sampleSize, pmqTable, vicinitySizes);
hold on
plot(boundM, 'b');
bm1 = hygecdf( TrainErrorOverfitThreshold(sampleSize, trainSize, [1:sampleSize], eps), sampleSize, ... 
                [1:sampleSize], trainSize)
plot(bm1, 'g');
grid on
figure
plot(optimalQ, 'r')
grid on
figure
plot(boundV ./ boundM, 'b')
if any(abs(bm1 - boundV) > 1e-9)
    'ERROR'
end


