function [sample, sampleClasses] = GenTapes(L)
x1 = [0 : 10 / L : 5 - 10/L];
y1 = 4 * sin(x1);
y2 = 7 * sin(x1);

sample = zeros(L, 2);

sampleClasses  = [ ones(1, L/2) (-1 * ones(1, L/2) ) ];
  
sample(:, 1) = [x1 x1 + 0.1]  
sample(:, 2) = [y1 y2]
end




