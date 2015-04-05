function BestRules(task, rules, p,n,eV, info)
%Создает рисунок с лучшими правилами по разным критериям

%terms = Calibrate(task);
%rules = RuleSetGeneratorTEMP(task, terms, []);

%[~, ~, p, n, eV, ~] = CalcRulesPNpn(rules, terms, task);

%params.fInfo = @HInfoD;
%info = CalcRulesInfo(rules, terms, task, params);

errorType1 = zeros(length(rules.target),1);
for i=1:length(rules.target)
    errorType1(i) = sum(eV(i,:) & (rules.target(i)==task.target)');
end;

errorType2 = zeros(length(rules.target),1);
for i=1:length(rules.target)
    errorType2(i) = sum(eV(i,:) & (rules.target(i)~=task.target)');
end;

infEps = bsearch(@(eps)(sum(info>eps)), 5000, 60);
inf = info>infEps;
v = sort(errorType1(rules.target==1));
t11Eps = v(2500);
v = sort(errorType1(rules.target==2));
t12Eps = v(2500);
t1 = (errorType1<t11Eps & rules.target==1) | (errorType1<t12Eps & rules.target==2);
v = sort(errorType2(rules.target==1));
t21Eps = v(2500);
v = sort(errorType2(rules.target==2));
t22Eps = v(2500);
t2 = (errorType2<t21Eps & rules.target==1) | (errorType2<t22Eps & rules.target==2);
figure;
hold on;
plot(p(inf),n(inf),'r.');
plot(p(t1),n(t1),'b.');
plot(p(t2),n(t2),'g.');
hold off;
end