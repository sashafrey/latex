function [sample, sampleClasses] = GenerateStrips(L, m)
    n1 = L/2 - m;
    n2 = L /2;
    n3 = m ;
    sample = [ [1:n1]' + 0.5 * rand(n1, 1) rand(n1, 1);  [1:n2]' + 0.5 * rand(n2, 1) 2 + rand(n2, 1);  [1:n3]' + 0.5 * rand(n3, 1) 4 + rand(n3, 1)  ];
    sampleClasses = [ones(1, n1)  -ones(1, n2) ones(1, n3)];
    %figure
    %scatter(sample(:, 1), sample(:, 2), 10, sampleClasses, 'filled');
end