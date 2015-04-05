function [QEps, avg, stddev, QEpsValue, allOverfittings] = CalcOverfitting(algset, trainRatio, itersCount, eps, maxEps, eps_step)
    if (~exist('maxEps', 'var'))
        maxEps = 1;
    end
    
    if (~exist('eps_step', 'var'))
        eps_step = 0.01;
    end
    
    algset = AlgsetGet(algset, 1:algset.Count);
    nItems = size(algset, 2);
    trainSize = floor(nItems * trainRatio);
    testSize = nItems - trainSize;
    koeff = 1 / trainSize + 1 / testSize;
    C = 0;

    if (trainSize == 0 || testSize == 0)
        throw(MException('InvalidArgument', 'Check trainRatio parameter - train or test sample turns out to be empty.'));
    end

    nTotalErrors = sum(algset, 2);

    nRepeats = 10;
    avg = zeros(10, 1);
    allOverfittings = VectorCreate();
    for repeat = 1:nRepeats
        overfittings = VectorCreate();
        for iter = 1:itersCount
            trainSample = randsample(nItems, trainSize);
            nTrainErrors = sum(algset(:, trainSample), 2);
            bitmask = (min(nTrainErrors) == nTrainErrors);
            ids = find(bitmask);
            [~, id] = max(nTotalErrors(bitmask));
            algId = ids(id);
            overfitting = nTotalErrors(algId) / testSize - nTrainErrors(algId) * koeff;
            overfittings = VectorAdd(overfittings, overfitting);
            C = C + (nTotalErrors(algId) - nTrainErrors(algId));
        end

        C = C / itersCount;    

        overfittings = VectorTrim(overfittings);
        overfittings = overfittings.Data;
        allOverfittings = VectorAdd(allOverfittings, overfittings);
        avg(repeat) = mean(overfittings);
    end
    
    allOverfittings = VectorTrim(allOverfittings);
    allOverfittings = allOverfittings.Data;
        
    QEps.X = (0:eps_step:maxEps)';
    nPoints = length(QEps.X);
    QEps.Y = zeros(nPoints, 1);
    for i=1:nPoints
        QEps.Y(i) = sum(allOverfittings >= QEps.X(i)) / (itersCount * nRepeats);
    end
    
    stddev = std(avg) / sqrt(10);
    avg = mean(avg);
    
    if (nargin < 5)
        eps = avg;
    end
    
    QEpsValue = QEps.Y(find(QEps.X >= eps, 1));
end
