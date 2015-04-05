function p=DrawLinks(p)
if (p.n~=2)
    printf('Данная функция поддерживает только двумерные выборки!')
    return
end;

if (exist('p.graph','var'))
    p=CalcLinks(p);
end;

for i=1:size(p.graph,1)
    for j=p.graph{i,1}
        plot([p.coord(j,1) p.coord(i,1)],[p.coord(j,2) p.coord(i,2)],'k-');
    end;
end;

axis([-0.5 p.L+0.5 -0.5 p.L+0.5]);
end