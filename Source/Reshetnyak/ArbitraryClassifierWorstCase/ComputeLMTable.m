function [lmTable] = ComputeLMTable(maxSampleSize)
    chooseTable = ComputeChooseTable(maxSampleSize);
    lmTable = zeros(maxSampleSize, floor(maxSampleSize/2) );
    lmTable(:, 1) = 1;
    for sampleSize = 2:maxSampleSize
        trainSize = ceil(sampleSize/ 2);
        for m = 2:floor(sampleSize/2)
            numAlgs = floor(sampleSize / m);
           
            negativeSum = 0;
            positiveSum = 0;
            flag = true;
            for i = 1:numAlgs
                if (flag) 
                    positiveSum = positiveSum  + chooseTable(numAlgs + 1, i + 1) *  ...
                    chooseTable(sampleSize - m * i + 1, trainSize + 1);
                else
                    negativeSum = negativeSum  + chooseTable(numAlgs + 1, i + 1) *  ...
                    chooseTable(sampleSize - m * i + 1, trainSize + 1);
                end
                flag = ~flag;
            end
            lmTable(sampleSize, m) = (positiveSum - negativeSum) / chooseTable(sampleSize + 1, trainSize + 1);
        end
       
    end

end