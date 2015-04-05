function [res] = ComputeExtendedSCBound(prmqTable, scProfile)
    res = sum( sum( sum( prmqTable(1:size(scProfile, 1), 1:size(scProfile, 2), 1:size(scProfile, 3)) .* scProfile ) ) );
end