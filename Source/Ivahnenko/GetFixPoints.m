%расчитать фиксированные точки для вборки.
%расчет ведется по графу связей.
%результат записывает в p.fixPoints
%{:,1} - фиксированные обьъекты на обучении
%{:,2} - фиксированные обьъекты на контроле

function p = GetFixPoints(p)

fixPoints = cell(length(p.graph),2);
[~, ix] = sort(p.errorsCount);
for i=ix'
    lmy = p.errors{i};
    
    childs = p.graph{i,1};
    fp = [];
    for j=childs
        lchild = p.errors{j};
        x = setxor(lmy, lchild);
        fp = union(fp, x);
    end;
    fixPoints{i,1} = sort(fp);
    
    parents = p.graph{i,2};
    fp = [];
    for j=parents
        lparent = p.errors{j};
        fp = union(fp, fixPoints{j,2});
        x = setxor(lmy, lparent);
        fp = union(fp, x);
    end;
    fixPoints{i,2} = sort(fp);
end;
% [~, ix] = sort(p.errorsCount,'descend');
% for i=ix'
%     parents = p.graph{i,2};
%     for j=parents
%         fp = fixPoints{j,1};
%         for k=fixPoints{j,1};
%             if (p.sample(k,end)==1)
%                 fp = setdiff(fp,k);
%             end;
%         end;
%         fp = setdiff(fp,fixPoints{i,2});
%         fixPoints{i,1} = union(fp,fixPoints{i,1});
%     end;
% end;

for i=ix'
    l1 = length(fixPoints{i,1});
    l2 = length(fixPoints{i,2});
    if (l1>p.L/2)
        fixPoints{i,1} = NaN;
        fixPoints{i,2} = NaN;
    elseif(l1==p.L/2)
        fixPoints{i,2} = setdiff(1:p.L,fixPoints{i,1});
    elseif(l2>p.L/2)
        fixPoints{i,1} = NaN;
        fixPoints{i,2} = NaN;
    elseif (l2==p.L/2)
        fixPoints{i,1} = setdiff(1:p.L,fixPoints{i,2});
    end;
end;

p.fixPoints = fixPoints;
end