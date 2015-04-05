function [sample, sampleClasses] = GenerateNormalClasses(numClasses, sampleSize, dim);
   if nargin < 3
       dim = 2;
   end
   
   alpha = 2 * pi / numClasses;
   classSize = floor(sampleSize / numClasses);
   R = 10;
   variance = ( sqrt(2 *R^2 - 2 * R^2* cos(alpha) ) / 6) ^2;
   sample = zeros(0, dim);
   sampleClasses = [];
   for i = 1:numClasses
       sample =  [sample; mvnrnd( repmat([R * cos(alpha * i) R * sin(alpha * i) zeros(1, dim - 2)],classSize, 1), ...
           variance * eye(dim) ) ];
       sampleClasses = [sampleClasses; i * ones(classSize, 1) ]; 
   end
   
end