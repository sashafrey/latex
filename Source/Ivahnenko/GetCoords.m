function [coord, map] = GetCoords(sample, varargin)
% размерность задачи
n = size(sample,2);
if (isempty(varargin))
    L = size(sample,1);
else
    L = varargin{1};
end;
% карта связей, число в ячейке i_1,i_2,..i_k отражает какие связи
% есть у данной клетки с остальными
fullLink = uint8(2^n-1);
map = ones((ones(1, n)*(L+1)),'uint8')*fullLink;

%разрываем связи
for i=0:size(sample,1)
    if (i==0)
        tmp = zeros(L,1);
    else
        tmp = sample(i,:);
    end;
    for dim=1:n
        s='';
        for j=1:n
            if (j==dim)
                s = [s,sprintf('tmp(%d)+1,',j)];
            else
                s = [s,sprintf('tmp(%d)+1:end,',j)];
            end;
        end;
        s = s(1:end-1);
        s1 = sprintf('map(%s)=bitset(map(%s),dim,0);',s,s);
        eval(s1);
    end;
end;
idx = find(map==0);
if (n==1)
    coord = idx;
elseif (n==2)
    [x1, x2] = ind2sub(size(map),idx);
    coord = sortrows([x1,x2]);
elseif (n==3)
    [x1, x2, x3] = ind2sub(size(map),idx);
    coord = sortrows([x1,x2,x3]);
elseif (n==4)
    [x1, x2, x3, x4] = ind2sub(size(map),idx);
    coord = sortrows([x1,x2,x3,x4]);
elseif (n==6)
    [x1, x2, x3, x4, x5, x6] = ind2sub(size(map),idx);
    coord = sortrows([x1,x2,x3,x4, x5, x6]);
elseif (n==5)
    [x1, x2, x3, x4, x5] = ind2sub(size(map),idx);
    coord = sortrows([x1,x2,x3,x4, x5]);
end;
coord = coord-1;
end