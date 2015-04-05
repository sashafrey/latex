function q = CalcAllQ(p)
pP = CalcLinks(p,'p');
pN = CalcLinks(p,'n');
p = CalcLinks(p);
q.Q = CalcQ(p,100);
q.Qp = CalcQ(pP,100);
q.Qn = CalcQ(pN,100);
%p = GetFixPoints(p);
%q.MyQ = CalcMyQ2(p);
%q.VQ = CalcVapnukQ(p);
end