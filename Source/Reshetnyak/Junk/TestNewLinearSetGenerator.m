%mex -output D:\Study\Science\Ёксперименты\AlgorithmSetsBuilders\BuildLinearSet D:\Study\Science\эксперименты\AlgorithmSetsBuilders\BuildLinearSet\build_linear_set.cpp
%[sample, sampleClasses] = GenerateCloseClasses(200);
[sample, sampleClasses] = GenerateSimpleSample(100, 3);
[algs, graph] = BuildLinearSet(sample, sampleClasses);

%PaintAlgorithmsFamily(algs, 10);
%scProfile = ComputeScProfile(1, BuildFamilyGraph(algs));
%b = BuildLinearSet(sample, sampleClasses);
%find(sortrows(a) ~= sortrows(b))