%% Заполняются поля graph, errors, errorsCount
% Необязательный аргумент - тип объектов
% b(oth = default), p(os), n(eg)
function p = CalcLinks(p, varargin)

if (p.n ~= 2)
    fprintf('Данная функция поддерживает только двумерные выборки!\n')
    return
end;

type = 'b';
if (~isempty(varargin))
    type = varargin{1};
end;

clusters = zeros(p.L+1);
%Ищем кластеры
if (exist('p.map','var')==0)
    if (type=='p')
        sample = p.sample(p.sample(:,end)==1,1:end-1);
        zErrors = find(p.sample(:,end)==1);
    elseif(type=='n')
        sample = p.sample(p.sample(:,end)==0,1:end-1);
        zErrors = [];
    else
        sample = p.sample(:,1:end-1);
        zErrors = find(p.sample(:,end)==1);
    end;
    [p.coord, p.map] = GetCoords(sample, p.L);
end;
z = size(p.coord,1);
for i=1:z
    clusters(p.coord(i,1)+1,p.coord(i,2)+1)=i;
end;

p.clusters = clusters;

%делаем граф
graph = cell(z,2);
errors = cell(z,1);
errors{1} = zErrors;
aP = zeros(z,1);%Флаг обработки кластера
aP(1) = 1;
queue = 1;
while ~isempty(queue)
    i = queue(1);
    rMax = p.L;
    for c=p.coord(i,2):p.L
        for r=p.coord(i,1):rMax
            if (c==p.coord(i,2) && r==p.coord(i,1)), continue;end;
            neib = p.clusters(r+1,c+1);
            if (neib~=0)
                rMax = r-1;
                %graph{i,1} = [graph{i,1}, neib];
                %graph{neib,2} = [graph{neib,2}, i];
                if (aP(neib)==0)
                    if (r==p.coord(i,1))
                        f = find(p.sample(:,2)==c,1,'first');
                    else
                        f = find(p.sample(:,1)==r,1,'first');
                    end;
                    errors{neib} = addError(f, errors{i});
                    queue = [queue, neib];
                    aP(neib)=1;
                end;
                if (length(errors{neib})>length(errors{i}))
                    graph{i,1} = [graph{i,1}, neib];
                    graph{neib,2} = [graph{neib,2}, i];
                else
                    graph{i,2} = [graph{i,2}, neib];
                    graph{neib,1} = [graph{neib,1}, i];
                end;
                break;
            end;
        end;
    end;
    queue = queue(2:end);
end;

p.graph = graph;
p.errors = errors;
p.errorsCount = zeros(z,1);
for i=1:length(errors)
    p.errorsCount(i) = length(errors{i});
end;
%сортируем по числу ошибок
% [p.errorsCount ix] = sort(errorsCount);
% for i=1:z
%     for j=1:length(p.graph{i,1})
%         p.graph{i,1}(j) = find(ix==p.graph{i,1}(j),1,'first');
%     end;
%     for j=1:length(p.graph{i,2})
%         p.graph{i,2}(j) = find(ix==p.graph{i,2}(j),1,'first');
%     end;
% end;
% for i=1:length(p.clusters(:))
%     if (p.clusters(i)~=0)
%         p.clusters(i) = find(ix==p.clusters(i),1,'first');
%     end;
% end;

function ret = addError(b, prevErrs)
    if (type=='p')
        ret = setdiff(prevErrs, b);
    elseif (type=='n')
        ret = sort([prevErrs; b]);
    else
        if (p.sample(b, end)==0)
            ret = sort([prevErrs; b]);
        else
            ret = setdiff(prevErrs, b);
        end;
    end;
end
end