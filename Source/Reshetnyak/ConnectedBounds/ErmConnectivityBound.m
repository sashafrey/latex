function [result] = ErmConnectivityBound(sampleSize, pmqTable, vicinitySizes)
    maxBound = zeros(1, max(vicinitySizes) + 1);
    for connectivity = 0 : numel(maxBound) - 1
        for  m = 1 : sampleSize
            for q = max(m + connectivity - sampleSize, 0) : min(connectivity, m) 
                maxBound(connectivity + 1) = max(maxBound(connectivity + 1), ...
                                            pmqTable(m - q + 1, m + 1, connectivity - q + 1));
            end
        end
    end
    
    result = sum(maxBound(vicinitySizes + 1));

end