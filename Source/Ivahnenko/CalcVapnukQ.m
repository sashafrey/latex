function Qeps = CalcVapnukQ(p)
l=p.L/2;
Qeps = zeros(2*l+1,2);
eps = -1:1/l:1;
Qeps(:,1) = eps;
%вероятности
Peps = zeros(length(p.errors),1);
% теперь считаем для каждого алгоритма его вклад в сумму Qeps
for aIdx=1:length(p.errors)
    %Число ошибок которое мы еще можем совершить
    m_a = double(p.errorsCount(aIdx));
      
    s_a = l/p.L*(double(p.errorsCount(aIdx))-eps*(p.L-l));
    h = H(s_a, p.L, m_a, l);
    Qeps(:,2) = Qeps(:,2)+h;
end;
end

%% Вычисление H(L,m,l)
function Heps=H(s,L,m,l)
epsLength = length(s);
Heps = zeros(epsLength,1);
s0 = max(0,m-(L-l));
s1 = min(m,l);
while (1==1)
    idx = find(s>s0, 1, 'last' )+1;
    if(isempty(idx) || s0>s1), break; end;
    Heps(idx) = Heps(idx) + nchoosek(m,s0)*nchoosek(L-m,l-s0)/nchoosek(L,l);
    s0 = s0+1;
end;

end
