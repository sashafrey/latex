function q = CalcQMod(p,N)
r = RandStream('mt19937ar','Seed',1);
l = p.L/2;
Qeps = zeros(2*l+1,2);
Qeps(:,1) = -1:1/l:1;
q.Qb = Qeps;
q.Qp = Qeps;
q.Qn = Qeps;
q.Info = zeros(N,1);
for i=1:N
    ix = r.randperm(p.L);
    trainSet = ix(1:l);
    testSet = ix(l+1:end);
   
    q1 = CalcOneStep(p,trainSet,testSet);
    
    q.Info(i) = q1.bestInfo;
    q.Qb(q1.bestDeltaBoth+l+1,2) = q.Qb(q1.bestDeltaBoth+l+1,2)+1;
    q.Qp(q1.bestDeltaPos+l+1,2) = q.Qp(q1.bestDeltaPos+l+1,2)+1;
    q.Qn(q1.bestDeltaNeg+l+1,2) = q.Qn(q1.bestDeltaNeg+l+1,2)+1;
end;

i = find(q.Qb(:)>0);
i = setdiff(i,1:2*l+1);
q.Qb(i) = q.Qb(i)/N;

i = find(q.Qp(:)>0);
i = setdiff(i,1:2*l+1);
q.Qp(i) = q.Qp(i)/N;

i = find(q.Qn(:)>0);
i = setdiff(i,1:2*l+1);
q.Qn(i) = q.Qn(i)/N;
end