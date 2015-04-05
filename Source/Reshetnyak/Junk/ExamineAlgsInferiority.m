sampleSize = 60;
trainSize = sampleSize / 2;
%[sample, sampleClasses] = GenerateCloseClasses(sampleSize);
[sample, sampleClasses] = GenerateSimpleSampleWithRandomNoise(sampleSize, 2, 4);
algs = BuildLinearSet(sample, sampleClasses);
PaintAlgorithmsFamily(algs, 10);
PaintSample(sample, sampleClasses);
[profile, inferiorObjects] = ComputeInferiorityProfile(algs);
profile = sum(profile, 3);

figure
colormap(1 - 0.8*gray);
%graymon
%surf(scProfile);
surf(0:size(profile, 2) - 1, 0:(size(profile, 1) - 1), profile)