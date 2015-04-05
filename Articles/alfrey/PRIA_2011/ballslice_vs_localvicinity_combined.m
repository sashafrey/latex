%% PIC1
L = 200;
ell = 100;
m = 50;

eps1=[];eps2=[];
rhoMax = 25;
for rho=1:rhoMax
    r = 2 * rho;
    PrepareNSO = NoisedSetOverfittingPrepare(L, ell, m - rho, r, rho, -1);
    eps1(rho) = bsearch(@(eps)NoisedSetOverfittingCalc(PrepareNSO, eps), 0.5, 1);
    eps2(rho) = BallSliceSubsetOverfitting(L, ell, m, r, -1);
    fprintf('%.4f, %.4f\n', eps1(rho), eps2(rho))
end

%% PIC2
L = 200;
ell = 100;
m = 50;
rho = 5;
r = 2 * rho;

QEps=[];QEps2=[];
dMax = 20;
%for d=[1,2,4,8,16,32,64,128,256]
for d=1:dMax
    PrepareNSO = NoisedSetOverfittingPrepare(L, ell, m - rho, r, rho, d);
    QEps(d) = NoisedSetOverfittingCalc(PrepareNSO, 0.1);
    [~, QEps2(d)] = BallSliceSubsetOverfitting(L, ell, m, r, d, 0.1);
    fprintf('%.4f %.4f\n', QEps(d), QEps2(d))
end
QEps2(1) = QEps(1);


%% Plot
style1 = 'k.-'; %'ks-';
style2 = 'k.--'; %'kv-';
drawLegend = false;

subplot(1,2,1)
plot(1:rhoMax, eps2, style1, 1:rhoMax, eps1, style2);
axis tight
if (drawLegend)
    legend('Ball slice', 'Local vicinity', 'location', 'best')
    legend boxoff
end
xlabel('\rho')
ylabel('\delta')


subplot(1,2,2)
plot(1:dMax, QEps2, style1, 1:dMax, QEps, style2)
if (drawLegend)
legend('Ball slice', 'Local vicinity', 'location', 'best')
legend boxoff
end
axis tight
    xlabel('d')
    ylabel('Q_\epsilon')

    