bound = zeros(1, 20);
for n = 1 : 20
    n
    bound(n) = ComputeGeneralizedGraphBoundTest(algs, graph, 0.13, trainSize, n);
end
bound
plot(bound);
    