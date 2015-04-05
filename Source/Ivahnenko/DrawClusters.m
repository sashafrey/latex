% Рисует классы эквивалентности.
function p=DrawClusters(p)
if (p.n~=2)
    printf('Данная функция поддерживает только двумерные выборки!')
    return
end;
if (exist('p.map','var')==0)
    [p.coord, p.map] = GetCoords(p.sample(:,1:end-1));
end;

for i=1:p.L+1
    for j=1:p.L+1
        plot(i-1,j-1,'.r', 'MarkerSize',10);
        if (bitget(p.map(i,j),1)==1)
            plot([i-1,i-2],[j-1,j-1],'-r','LineWidth',2);
        end;
        
        if (bitget(p.map(i,j),2)==1)
            plot([i-1,i-1],[j-1,j-2],'-r','LineWidth',2);
        end;
    end;
end;

axis([-0.5 p.L+0.5 -0.5 p.L+0.5]);
end