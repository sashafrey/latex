%% Функция создания выборки
% Ожидаемые параметры:
% p.L - длинна выборки
% p.n - размерность пространства
% p.seed - инициализация генератора случайных чисел
function p = CreateCorrect1(p,n)
    r = p.rand;
    target = p.sample(:,end);
    iC0 = target==0;
    iC1 = target==1;
    sample = zeros(p.L,n);
    d = 0.5^(1/n);
    sample(iC1,:) = r.rand(sum(iC1),n)*d;
    for i=find(iC0)'
        v = r.rand(1,n);
        while (all(v<d))
            v = r.rand(1,n);
        end;
        sample(i,:) = v;
    end;
    p.sample = [p.sample(:,1:end-1), sample, target];
    p.n = p.n + n;
end