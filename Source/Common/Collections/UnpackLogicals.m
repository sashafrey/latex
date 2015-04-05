function logicals = UnpackLogicals(packedLogicals, nCols, indexes)
%"unpackLogicals"
%   Decompresses a logical vector or array that was compressed using
%   packLogicals.  The size of the initial data must be known.
%
%By James R. Alaly
%
%JRA 7/16/04
%JRA 9/14/04 - Vectorize for speed improvement.
%JRA 12/27/04 - Stores size vector in data struct. Iram Weinstein's idea.
%
%Usage:
%   function logicals = unpackLogicals(data, siz)

if (~exist('indexes', 'var'))
    indexes = 1:size(packedLogicals, 1);
end
    
nRows = size(indexes, 1);
nUINTS = size(packedLogicals, 2);
logicals = false(nRows, nUINTS * 64);

for i=indexes
    for j=1:64
        logicals(i, j:64:end) = bitget(packedLogicals(i, :), j);   
    end
end

logicals(:, nCols + 1 : end) = [];

%---Old, Unvectorized Code.
% logicals = logical(zeros([1 nData]));
% bitNumV = mod((1:nData)-1, 8) + 1;
% uintNumV = floor(((1:nData)-1)/8) + 1;
% for i=1:nData
%     logicals(i) = bitget(data(uintNumV(i)), bitNumV(i));
% end
% logicals = reshape(logicals, siz);