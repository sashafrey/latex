function [prob] = ComputeOverfitProbabiltyForLexicographicallyFirstSet ...
    (numAlgs, sampleSize, trainSize, errorLevel, maxTrainError, chooseTable)
%Функции не передали таблицу биномиальных коэффициентов или передали
%таблицу неправильного размера
if (nargin < 5 || ( size(chooseTable, 1) < sampleSize + 1) )
    chooseTable = ComputeChooseTable(sampleSize);
end

prob = 0;
factor = 1;
while (numAlgs > 0 && trainSize > -1 && errorLevel > 0)
    fixedObjects = errorLevel;
    while ( chooseTable(sampleSize - fixedObjects + 1, errorLevel - fixedObjects + 1) < numAlgs)
        fixedObjects = fixedObjects - 1;
    end
    if (chooseTable(sampleSize - fixedObjects + 1, errorLevel - fixedObjects + 1) > numAlgs)
        fixedObjects = fixedObjects + 1;
    end
    prob = prob + factor * HypergeometricTail(sampleSize, trainSize, fixedObjects, maxTrainError, chooseTable);
    if fixedObjects <= maxTrainError
        break;
    end
    numAlgs = numAlgs - chooseTable(sampleSize - fixedObjects + 1, errorLevel - fixedObjects + 1);
    sampleSize = sampleSize - fixedObjects;
    trainSize = trainSize - 1 - maxTrainError;
    errorLevel = errorLevel - fixedObjects + 1;
    factor = chooseTable(fixedObjects + 1, maxTrainError + 1);
    maxTrainError = 0;
end

prob = prob /  chooseTable(sampleSize + 1, trainSize + 1);

end