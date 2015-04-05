function [bound, optSource] = TransductivePessimisticERM(algs, trainSample, trainSampleClasses, eps)

%     function [res] = ComputeSCBound(scProfile)
%         res = sum( sum( pmqTable(1:size(scProfile,1), 1:size(scProfile, 2)) .* scProfile ) );
%     end

    function [res] = ComputeExpectedRisk(scProfile)
        size(scProfile)
        res = sum( sum( choiceProbTable(1:size(scProfile,1), 1:size(scProfile, 2)) .* scProfile ) ) / ...
          (sampleSize - numel(trainSample)) ;
    end

    familyGraph = BuildFamilyGraph(algs);
    [numAlgs, sampleSize] = size(algs);
    %overfitLevel = ceil((sampleSize - numel(trainSample)) * eps)
    numAlgs
    for pos = 1 : numel(eps)
        pmqTable(:, :, pos) = ComputePmqTable(sampleSize, numel(trainSample), eps(pos), 0, sampleSize);
    end
    %choiceProbTable = ComputeAlgorithmChoiceProbTable(sampleSize, numel(trainSample), 0, sampleSize);
    trainSampleClasses(trainSampleClasses == -1) = 0;
    bound = [inf(size(eps)); zeros(size(eps))];
    optSource = zeros(size(bound));
    potentialSources = 0;
    expectedRisk =[];
    for n = 1:numAlgs
        if all(algs(n, trainSample) == trainSampleClasses)
            [scProfile, isOneSource] = ComputeScProfile(n, familyGraph);
            if size(scProfile, 1) == sampleSize + 1 && isOneSource
                potentialSources = potentialSources + 1;
                for pos = 1 : numel(eps)
                    curBound = ComputeSCBound(pmqTable(:, :, pos), scProfile);
                    if bound(2, pos) < curBound
                        optSource(2, pos) = n;
                        bound(2, pos) = curBound;
                    end
                    if bound(1, pos) > curBound
                        optSource(1, pos) = n;
                        bound(1, pos) = curBound;
                    end
                end
                %expectedRisk(potentialSources) = ComputeExpectedRisk(scProfile);
            end
        end
    end
    potentialSources
    %min(expectedRisk)
    %max(expectedRisk)
end