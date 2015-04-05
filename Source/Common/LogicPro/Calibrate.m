function terms = Calibrate(task, nMax)
    % nMax:
    %   a) For nominal features - #terms <= (nMax + 1)
    %   b) For numerical features - #terms <= 0.5 * nMax*(nMax+1) + 1.
    %
    % output - table struct with the following fields:
    %   feature, 
    %   left, 
	%   right, 
    %   isnot

    if (~exist('nMax', 'var'))
        nMax = 10;        
    end
    
    terms.feature = [];
    terms.left = [];
    terms.right = [];
    terms.isnot = [];
    
    for iFeature=1:task.nFeatures
        values = task.objects(:, iFeature);
        
        if (any(isnan(values)))
            terms = tsConcat(terms, CreateTerm(iFeature, NaN, NaN));
        end
        
        uniquevalues = unique(values);
        uniquevalues(isnan(uniquevalues)) = [];
        
        if (task.isnominal(iFeature))
            if (length(uniquevalues) > nMax)
                %select most frequent values.
                nItemsPerValue = zeros(length(uniquevalues), 1);
                for iValue = 1:length(uniquevalues)
                    nItemsPerValue(iValue) = sum(values == uniquevalues(iValue));
                end
                
                [~, idx] = sort(nItemsPerValue, 'descend');
                uniquevalues = uniquevalues(idx(1:nMax));
            end
            
            for value = uniquevalues'
                terms = tsConcat(terms, CreateTerm(iFeature, value, value));
            end
        else
            [thresholds, counts, targets] = CreateInitialThresholds(uniquevalues, iFeature, task);
            [thresholds, counts] = MergeByTarget(thresholds, counts, targets);
            
            nNotIsNan = sum(~isnan(task.objects(:, iFeature)));
            Check(sum(counts) == nNotIsNan);
            
            % Use two alternative methods and select the one with smaler std
            [thresholds1, counts1] = MergeSmallUniformIntervals(thresholds, counts, nMax);
            [thresholds2, counts2] = MergeSmallestIntervals(thresholds, counts, nMax);
            
            % Self-testing.
            Check(sum(counts1) == nNotIsNan);
            Check(sum(counts2) == nNotIsNan);
            
            if (std(counts1) < std(counts2))
                thresholds = thresholds1;
            else
                thresholds = thresholds2;
            end
            
            for i=0:length(thresholds)
                if (i==0) 
                    left = -inf; 
                else
                    left = thresholds(i);
                end;
                
                for j=(i+1):length(thresholds)
                    terms = tsConcat(terms, CreateTerm(iFeature, left, thresholds(j)));
                end
            end
        end
    end
    
    Check(tsIsValid(terms));
end

function term = CreateTerm(iFeature, left, right)
    term.feature = iFeature;
    term.isnot = false(length(iFeature), 1);
    term.left = left;
    term.right = right;
end

function [thresholds, counts, targets] = CreateInitialThresholds(uniquevalues, iFeature, task)
    %counts(i) holds the number of objects in the interval [thresholds(j-1), thresholds(j)]
    %where the semantic of Thresholds(0) is "-inf".
    thresholds = [0.5 * (uniquevalues(1:end-1) + uniquevalues(2:end)); +inf];
    nThresholds = length(thresholds);
    
    left = [-inf; thresholds(1:(nThresholds - 1))];
    right = thresholds;    
    
    iFeatures = ones(nThresholds, 1) * iFeature;
    coverage = CalcTermsCoverage(CreateTerm(iFeatures, left, right), task);
    counts = sum(coverage, 2);

    targets = NaN(nThresholds, 1);
    for i=1:nThresholds
        curTargets = unique(task.target(coverage(i, :)));
        if (length(curTargets) == 1)
            targets(i) = curTargets;
        end
    end
end

function [thresholds, counts] = MergeByTarget(thresholds, counts, targets)
    nThresholds = length(thresholds);
    
    newThresholds = [];
    newCounts = [];
    cumulativeCount = 0;
    for i = 1:(nThresholds-1)
        cumulativeCount = cumulativeCount + counts(i);
        if (targets(i) == targets(i+1)) % this automatically handles NaN
            continue;
        end
        
        newThresholds(end + 1) = thresholds(i);
        newCounts(end + 1) = cumulativeCount;
        cumulativeCount = 0;
    end
    
    newThresholds(end + 1) = thresholds(nThresholds);
    newCounts(end + 1) = cumulativeCount + counts(nThresholds);
    
    thresholds = newThresholds;
    counts = newCounts;
end

function [thresholds, counts] = MergeSmallUniformIntervals(thresholds, counts, targetLen)
    if (length(counts) <= targetLen)
        return;
    end
    
    remainder = sum(counts);
    iThreshold = 1;
    
    newThresholds = zeros(1, targetLen);
    newCounts = zeros(1, targetLen);
    for i = 1:targetLen
        toInclude = floor(remainder / (targetLen - i + 1));
        firstInterval = true;
        while((iThreshold <= length(counts)) && (firstInterval || (toInclude >= counts(iThreshold))))
            firstInterval = false;
            toInclude = toInclude - counts(iThreshold);
            remainder = remainder - counts(iThreshold);
            newCounts(i) = newCounts(i) + counts(iThreshold);
            newThresholds(i) = thresholds(iThreshold);
            iThreshold = iThreshold + 1;
        end
    end
    
    thresholds = newThresholds;
    counts = newCounts;
end

function [thresholds, counts] = MergeSmallestIntervals(thresholds, counts, nMax)
    while(length(thresholds) > nMax)
        [~, minId] = min(counts);
        if (minId == 1)
            mergeTo = 2;
        elseif (minId == length(counts))
            mergeTo = length(counts) - 1;
        else
            if (counts(minId + 1) < counts(minId - 1))
                mergeTo = minId + 1;
            else
                mergeTo = minId - 1;
            end
        end

        counts(mergeTo) = counts(mergeTo) + counts(minId);
        counts(minId) = [];
        thresholds(minId) = [];
    end
end