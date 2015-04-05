function [packedLogicals, nCols] = PackLogicals32(logicals)
%"packLogicals"
%   Compresses a logical vector or array so that each element takes up only
%   one bit instead of the usual 8 bits.  Unpack using unpackLogicals,
%   which requires the size of the original input.
%
%By James R. Alaly
%
%JRA 7/15/04  - Created.
%JRA 12/22/04 - Vectorize for speed improvement.
%JRA 12/27/04 - Stores size vector in data struct. Iram Weinstein's idea.
%
%Usage:
%   function data = packLogicals(logicals)

if ~islogical(logicals)
    error('packLogicals.m only works on logical inputs.');
    return;
end

nRows = size(logicals, 1);
nCols = size(logicals, 2);
nUINTS = ceil(nCols/32);

packedLogicals = repmat(uint32(0), nRows, nUINTS);

for i=1:nRows
    for j=1:32
        thisBitValue = logicals(i, j:32:end);
        trueInds = find(thisBitValue);
        packedLogicals(i, trueInds) = bitset(packedLogicals(i, trueInds), j, 1);
    end
end

% %Old, nonvectorized code
% if ~islogical(logicals)
%     error('packLogicals.m only works on logical inputs.');
%     return;
% end
% 
% 
% nData = length(logicals(:));
% nUINTS = ceil(nData/8);
% 
% data = uint8(zeros([1 nUINTS]));
% 
% bitNumV = mod((1:nData)-1, 8) + 1;
% uintNumV = floor(((1:nData)-1)/8) + 1;
% for i=1:nData
%     data(uintNumV(i)) = bitset(data(uintNumV(i)), bitNumV(i), logicals(i));        
% end