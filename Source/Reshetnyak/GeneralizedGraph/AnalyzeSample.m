function AnalyzeSample(sample, sampleClasses)
    sampleSize = size(sample, 1);
    trainSize = floor(sampleSize / 2);
    [graph, algs] = BuildConjunctionSetNew(sample, sampleClasses, sampleSize + 1, 4, 20);
    for n = 1 : numel(graph)
        graph{n} = graph{n}';
    end
    PaintAlgorithmsFamily(algs, 10, true);
    edgesDistr = GetEdgesDistribution(algs, graph)
    minError = min(sum(algs, 2))
    profile = ComputeInferiorityProfile(algs, graph);
    eps = [0.01 : 0.02 : 0.19];
    boundSC = zeros(1, numel(eps));
    boundHasse = zeros(1, numel(eps));
    for n = 1 : numel(eps)
        n
        %pmqTable = ComputeExtendedPmqTable(sampleSize, trainSize, eps(n), size(profile, 3), min(sum(algs, 2)));
        %boundSC(n) = min(1, ComputeExtendedSCBound(pmqTable, profile));
        boundHasse(n) = min(1, ComputeGeneralizedGraphBoundTest(algs, graph, eps(n), trainSize));
    end
    figure
    hold on
    plot(eps, boundSC, 'b');
    plot(eps, boundHasse, 'r');
    plot(eps, boundHasse ./ boundSC, 'g')
    
end