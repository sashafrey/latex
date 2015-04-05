function q = CheckResult(q)
N = length(q.qC);
q.pC = zeros(N,1);
q.pN = zeros(N,1);
q.pC10 = zeros(N,1);
q.pN10 = zeros(N,1);
q.nC = zeros(N,1);
q.nN = zeros(N,1);
q.nC10 = zeros(N,1);
q.nN10 = zeros(N,1);
q.all = zeros(N,8);
for i=1:N
    q.nC(i) = q.qC(i).testNeg;
    q.nN(i) = q.qN(i).testNeg;
    q.nC10(i) = q.qC10(i).testNeg;
    q.nN10(i) = q.qN10(i).testNeg;
    
    q.pC(i) = q.qC(i).testPos;
    q.pN(i) = q.qN(i).testPos;
    q.pC10(i) = q.qC10(i).testPos;
    q.pN10(i) = q.qN10(i).testPos;
%     q.nC(i) = q.qC(i).acTest-q.qC(i).rcTest;
%     q.nN(i) = q.qN(i).acTest - q.qN(i).rcTest;
%     q.nC10(i) = q.qC10(i).acTest - q.qC10(i).rcTest;
%     q.nN10(i) = q.qN10(i).acTest - q.qN10(i).rcTest;
%     
%     q.pC(i) = q.qC(i).rcTest;
%     q.pN(i) = q.qN(i).rcTest;
%     q.pC10(i) = q.qC10(i).rcTest;
%     q.pN10(i) = q.qN10(i).rcTest;
end
q.all(:,1) = q.nC;
q.all(:,2) = q.nN;
q.all(:,3) = q.nC10;
q.all(:,4) = q.nN10;

q.all(:,5) = q.pC;
q.all(:,6) = q.pN;
q.all(:,7) = q.pC10;
q.all(:,8) = q.pN10;

end