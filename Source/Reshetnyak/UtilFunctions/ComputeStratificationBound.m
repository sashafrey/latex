function [res] = ComputeStratificationBound(pmqTable, stratProfile)
    res = sum(stratProfile .* pmqTable(:,1));
end