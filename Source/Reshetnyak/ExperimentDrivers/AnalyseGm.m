function AnalyseGm(sampleSize, trainSize, eps)

    if nargin < 1
        sampleSize = 100;
    end
    if nargin < 2
        trainSize = sampleSize / 2;
    end
    if nargin < 3
        eps = 0.1;
    end
    
    pmqTable = ComputePmqTable(sampleSize, trainSize, eps, 5, 0) ;
    plot(pmqTable / nchoosek(sampleSize, trainSize));

end