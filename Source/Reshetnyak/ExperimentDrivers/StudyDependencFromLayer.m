sampleSize = 300;
trainSize = 150;
eps = 0.05;
% [sample, sampleClasses] = GenerateSimpleSample(sampleSize);
% algs = BuildLinearSet(sample, sampleClasses);
% 
% scProfile = ComputeSCProfile(1, BuildFamilyGraph(algs));
 probMQ = ComputePmqTable(sampleSize, trainSize, eps, 0, size(scProfile, 2) - 1);


%upperBound = probMQ(:, 1)' .* sum(scProfile, 2)';
lowerBound = zeros(1, sampleSize);
upperBound = zeros(1, sampleSize);

for m = 2 : sampleSize
    lowerBound(m) = lowerBound(m - 1) + probMQ(m + 1, :) * scProfile(m + 1, :)';
    upperBound(m) = lowerBound(m - 1) + probMQ(m + 1, 1) * sum(scProfile(m + 1, :))
end

hold on
grid on
plot(lowerBound(1:30), 'b');
plot(upperBound(1:30), 'r');


