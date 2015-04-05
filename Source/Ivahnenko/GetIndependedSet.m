%% Получить набор независимых объектов по выборке. Выборка передается без
% классов
function [indSet] = GetIndependedSet(sample)
n=size(sample,2);
L=size(sample,1);
indSet = cell(nchoosek(L,n),2);
is=1;

Test(1,0,[]);

indSet = indSet(1:is-1,:);

function []= Test(startFrom, deep, testSet)
    if (deep>0)
        m = zeros(1,deep);
        [~,idx]=max(sample(testSet,:),[],1);
        for k=idx
            m(k) = m(k)+1;
        end;

        if (all(m>0))
            indSet{is,1}=testSet;
            indSet{is,2}=m;
            is = is+1;
        end;
    end;
    if (deep==n), return;end;
    
    for i=startFrom:L
        Test(i+1,deep+1, [testSet i]);
    end;
end

end