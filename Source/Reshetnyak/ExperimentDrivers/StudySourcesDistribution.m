sampleSize = 80;
%trainSize = 50;

sample = mvnrnd( repmat(0, sampleSize, 2), 3 * eye(2));
sampleClasses = rand(1, sampleSize);
sampleClasses(sampleClasses > 0.5) = 1;
sampleClasses(sampleClasses <= 0.5) = -1;





PaintSample(sample, sampleClasses);
algs = BuildLinearSet(sample, sampleClasses);
sourcesDistr = zeros(1, sampleSize + 1);

for i = 1 : size(algs,1)
    isSource = true;
    for j = 1 : i - 1
        if ( and(algs(i, :), algs(j, :)) == algs(j, :) )
            isSource = false;
            break;
        end
    end
    s = sum(algs(i, :)) + 1;
    sourcesDistr(s) = sourcesDistr(s) + isSource;
end
sourcesCount = sum(sourcesDistr);
PaintAlgorithmsFamily(algs, 5);
figure
plot([0 : sampleSize], sourcesDistr);