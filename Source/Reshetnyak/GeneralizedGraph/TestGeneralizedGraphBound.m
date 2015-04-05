sampleSize = 100;
trainSize = sampleSize / 2
for test = 1 : 10
    test
    ind = randsample(size(scored_reads, 1), sampleSize);
    sample = [scored_reads(ind, 2 : 3)];
    sampleClasses = scored_reads(ind, 1)';
    [graph, algs] = BuildConjunctionSetNew(sample, sampleClasses);
    for n = 1 : numel(graph)
        graph{n} = graph{n}';
    end
    res1 = ComputeGeneralizedGraphBound(algs, graph, 0.05, trainSize)
    res2 = ComputeGeneralizedGraphBoundTest(algs, graph, 0.05, trainSize)
    if res1 ~= res2
        'Error'
        test
        %PaintAlgorithmsFamily(newAlgs, 10);
        %PaintSample(sample, sampleClasses);
        %break;
    end
         
        
   
end