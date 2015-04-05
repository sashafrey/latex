sampleSize = 4;
[sample, sampleClasses] = GenerateSimpleSample(sampleSize);
PaintBorders(sample, sampleClasses);
figure
scatter(sample(:, 1), sample(:,2), 30, 'r', 'filled');