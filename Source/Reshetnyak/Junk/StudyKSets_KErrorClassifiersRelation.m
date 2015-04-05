sampleSize = 30
d = 30
ang = pi

[sample, sampleClasses] = GenerateTwoSeparableConvexClasses(sampleSize, ang, d);
algs = BuildLinearSet(sample, sampleClasses);
PaintSample(sample, sampleClasses);
PaintAlgorithmsFamily(algs);