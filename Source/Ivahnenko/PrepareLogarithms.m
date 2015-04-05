function p = PrepareLogarithms(p)
p.logs = zeros(2*p.L+1,1);
for i=1:p.L
    p.logs(i+1)=p.logs(i)+log(i);
end;
end