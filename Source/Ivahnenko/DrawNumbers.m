function p=DrawNumbers(p, type)
if (p.n~=2)
    printf('Данная функция поддерживает только двумерные выборки!')
    return
end;
if (strcmp('errCount', type))
    p=DrawErrorsCount(p);
elseif(strcmp('clustNumb', type))
    p=DrawClusterNumbers(p);
elseif(strcmp('pos', type))
    p = DrawPositiveNumbers(p);
end;
end

function p=DrawErrorsCount(p)
    if (exist('p.graph','var')==0)
        p = CalcLinks(p);
    end;

    for i=1:length(p.errors)
        text(p.coord(i,1)+0.25,p.coord(i,2)+0.25,num2str(p.errorsCount(i)));
    end;
end

function p=DrawClusterNumbers(p)
    if (exist('p.map','var')==0)
        [p.coord, p.map] = GetCoords(p.sample(:,1:end-1));
    end;
    % код рисующий исходную выборку
    for i=1:size(p.coord,1)
        text(p.coord(i,1)+0.25,p.coord(i,2)+0.25,num2str(i));
    end;
end

function p = DrawPositiveNumbers(p)
    if (exist('p.graph','var')==0)
        p = CalcLinks(p);
    end;
    for i=1:length(p.errors)
        a_f = repmat(p.coord(i,:),p.L,1);
        delta = p.sample(:,1:end-1)-a_f;
        cover = all(delta<=0, 2);
        rightCovered = sum(cover & (p.sample(:,end)==1));
        text(p.coord(i,1)+0.25,p.coord(i,2)+0.25,num2str(rightCovered));
    end;
end