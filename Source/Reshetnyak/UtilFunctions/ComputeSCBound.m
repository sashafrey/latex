function [res] = ComputeSCBound(pmqTable, scProfile)
    res = sum( sum( pmqTable(1:size(scProfile,1), 1:size(scProfile, 2)) .* scProfile ) );
end