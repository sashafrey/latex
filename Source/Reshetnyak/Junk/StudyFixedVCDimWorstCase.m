sampleSize = 16;
trainSize = 6;
errorLevel = 4;
eps = errorLevel / (sampleSize - trainSize) - 1e-6;
[sample, sampleClasses] = GenerateSimpleSample(sampleSize);
algs = BuildLinearSet(sample, sampleClasses);
PaintAlgorithmsFamily(algs);
overfitProb = ExactFunctional(algs, trainSize, eps)
upperBound = nchoosek(sampleSize - (errorLevel - 3), trainSize) / nchoosek(sampleSize, trainSize)