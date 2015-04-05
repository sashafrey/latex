%% Функция создания выборки
% Ожидаемые параметры:
% p.L - длинна выборки
% p.n - размерность пространства
% p.seed - инициализация генератора случайных чисел
function p = CreateCorrectN(p,n)
    r = p.rand;
    target = p.sample(:,end);
    iC0 = find(target==0);
    iC1 = find(target==1);
    count1 = sum(target);
    a = sqrt(count1/(p.L*(n+1)));% размер закономерности 2а х а
    sample = zeros(p.L,n);
    lastI = 1;
    c = -3;
    for i=linspace(1,count1, n+2)
        c = c + 1;
        if (i==1), continue; end;
        
        v = lastI:i;
        lastI = v(end)+1;
        x = r.rand(length(v),n)*a;
        if (c>=0)
            vec = length(v)*c+1:length(v)*(c+1);
            x(vec) = x(vec)+a;
        end;
        sample(iC1(v),:) = x;
    end;
    
    for i=iC0'
        v = r.rand(1,n);
        while (all(v<2*a) && any(v<a))
            v = r.rand(1,n);
        end;
        sample(i,:) = v;
    end;
    p.sample = [p.sample(:,1:end-1), sample, target];
    p.n = p.n + n;
end