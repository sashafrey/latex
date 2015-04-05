%% Функция создания выборки
% Ожидаемые параметры:
% p.L - длинна выборки
% p.n - размерность пространства
% p.seed - инициализация генератора случайных чисел
% lambda - коэффициент проникновения в область чужого класса.
function p = CreateNoisy1(p,n,lambda,noiseObject)
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

    p1 = p;
    p1.sample = [p.sample(:,1:end-1), sample, target];
    p1.n = p.n + n;
    
    %здесь генерируем шумы.
    %1. ищем расстоняние до границы закономерности
    dist = zeros(1,p.L);
    for i=1:p.L
        di = sample(i,:)-d;
        if (all(di<0))
            dist(i)=min(abs(di));
        else
            dist(i)=sum(di(di>=0));
        end;
    end;
    [~,ix] = sort(dist);
    movedObj = [];
    for i=1:noiseObject
        ras = exp(-lambda*(1:p.L));
        if (~isempty(movedObj)), ras(movedObj)=0; end;
        ras = ras/sum(ras);
        s=0; j=0;
        th = r.rand(1);
        while(s<th)
            j = j+1;
            s = s+ras(j);
        end;
        movedObj = [movedObj j];
        target(ix(j)) = 1-target(ix(j));
    end;
    
    p.sample = [p.sample(:,1:end-1), sample, target];
    p.n = p.n + n;
end