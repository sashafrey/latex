sampleSize = 200
%numAlgs = 200;
%algs = BuildNoChain(BuildStratifiedChain(sampleSize, 10, numAlgs));
m0 = 40; 
m1 = 40;
m2 = 0;

%algs = [true(1, m0 + m2) false(1, sampleSize - m0 - m2); ...
%        true(1, m2) false(1, sampleSize - m1 - m2) true(1,m1)];
%rres = OnlineMonteCarlo(algs, @IsErrorOnNextObject, 100000)
res = TwoAlgorithmsCCV(sampleSize, m0, m1, m2)
m = 10;
length = 5;
%algs = BuildMonotoneChain(sampleSize, m, length + 1);
%res = MonotoneChainCCV(sampleSize, m, length);
% monteCarlo = zeros(1, sampleSize - 1);
% for t = 1 : sampleSize - 1
%     [waste, monteCarlo(t)] = MonteCarloEstimation(algs, t, 0.05, 10000);
% end
% monteCarlo

hold on
grid on
%plot(monteCarlo)
%plot(res,'r')
plot([1 : sampleSize - 1], res, 'k', 'LineWidth', 2)


