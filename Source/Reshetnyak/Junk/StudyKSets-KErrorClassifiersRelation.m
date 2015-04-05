sampleSize = 20
d = 3
angle = pi

[sample, sampleClasses] = GenerateTwoSeparableConvexClasses(sampleSize, angle, d);
algs = BuildLinearSet(sample, sampleClasses);
PaintSample(sample, sampleClasses);
PaintAlgorithmsFamily(algs);