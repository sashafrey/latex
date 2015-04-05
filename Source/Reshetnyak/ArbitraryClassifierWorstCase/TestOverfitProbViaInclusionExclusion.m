
algs = [1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1;
        1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
        0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0;
        0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0];
%         0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0;
%         0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1;
%         1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0;
        
        
trainSize = 9;
m = 3
numAlgs = size(algs, 1);
 [prob, coeffs] = ComputeOverfitProbViaInclusionExclusion(algs, trainSize, m)
% 1 - (1 - 2^(- m) )^numAlgs
% %lmTable = ComputeLMTable(500);
% %PlotResultsTable(lmTable(:, 1 : 50) > 0.5, [0 200], [0 100], '', 'Sample size', 'Error Level');
% GreedyFindNextAlgorithm(algs, trainSize,m)
% %testProb = ExactFunctional(algs, trainSize, m / (size(algs, 2) - trainSize) );
% %testProb = testProb( numAlgs)
