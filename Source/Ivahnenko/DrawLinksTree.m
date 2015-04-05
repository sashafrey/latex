function p = DrawLinksTree(p)

[~, ix] = sort(p.errorsCount);
z = p.errorsCount;
z2 = zeros(max(p.errorsCount)+2,1);
%ищем максимальное число алгоритмов с одинаковым числом ошибок.
maxCount = 0;
prev = -1;
curCount = 0;
for i=ix'
    z2(p.errorsCount(i)+1) = z2(p.errorsCount(i)+1)+1;
    if (p.errorsCount(i)~=prev)
        maxCount = max(maxCount,curCount);
        curCount=0;
    end;
    curCount = curCount+1;
    z(i) = curCount;
    prev = p.errorsCount(i);
end;


fig1 = figure;
axes1 = axes('Parent',fig1,'YGrid','on','XTickLabel',{},'XTick',zeros(1,0));
box(axes1,'on');
axis([0.5 maxCount+0.5 min(p.errorsCount)-0.5 max(p.errorsCount)+0.5]);
hold(axes1,'all');
hold on;
for i=ix'
    x1 = (maxCount-z2(p.errorsCount(i)+1))/2;
    x2 = (maxCount-z2(p.errorsCount(i)+2))/2;
    for j=p.graph{i,1}
        plot([z(i)+x1, z(j)+x2], [ p.errorsCount(i), p.errorsCount(j)],'k-');
    end;
    f1 = p.coord(i,1); % номер первой точки
    if (f1==0),continue;end;
    if (all(p.coord(i,:)==p.sample(f1,1:end-1)))
        if (p.sample(f1,end)==1)
            plot(z(i)+x1,p.errorsCount(i),'*k','MarkerSize',10, 'LineWidth',2);
        else
            plot(z(i)+x1,p.errorsCount(i),'ok','MarkerSize',10, 'LineWidth',2);
        end;
    end;
end;
hold off;

end