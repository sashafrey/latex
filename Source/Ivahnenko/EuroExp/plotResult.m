function plotResult(res)
figure;
hold on;
plot(res.trainErr,'g-');
plot(res.testErr, 'r-')
tmp = res.trainErr;
tmp(1:end) = res.lTrainErr;
plot(tmp, 'g.-');
tmp(1:end) = res.lTestErr;
plot(tmp, 'r.-');
hold off;
end