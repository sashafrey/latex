%% ‘ункци€ замен€юща€ значение признака его пор€дковым номером.
function p = NormalizeSample(p)
for i=1:p.n
    [~, ix] = sort(p.sample(:,i));
    [~, ix] = sort(ix);
    p.sample(:,i) = ix;
end;
[~,ix] = sort(p.sample(:,1));
p.sample = p.sample(ix,:);

end