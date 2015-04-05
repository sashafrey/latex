function LinearSampleTransductiveErmExperiment
    [sample, sampleClasses] = GenerateSimpleSample(10);
    t = BuildLinearSet(sample, sampleClasses);
    size(t)
end