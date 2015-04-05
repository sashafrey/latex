function maxAlgCount=CheckAllSamples(L)
sample = zeros(L,3);
teorMax = nchoosek(L,3)+nchoosek(L,2)+L;
fprintf(1,'TeorMax: %d\n',teorMax);
sample(:,1) = 1:L;
v = (1:L)';
ps = perms(1:L);
maxAlgCount = 0;
for i=ps'
    sample(:,2)=i;
    %for j=(L:-1:1)'
    for j=ps'
        sample(:,3)=j;
        [coords ~] = GetCoords(sample);
        s = size(coords,1);
        if (maxAlgCount<s)
            maxAlgCount = s;
            disp(maxAlgCount);
        end;
        %maxAlgCount = max(maxAlgCount,size(coords,1));
        if (maxAlgCount==teorMax)
            sample
            return;
        end;
    end;
end;
end