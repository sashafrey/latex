%Рисует точки выборки
function p=DrawPoints(p)
idx0 = find(p.sample(:,end)==0);
idx1 = find(p.sample(:,end)==1);
plot(p.sample(idx0,1),p.sample(idx0,2),'ob','MarkerSize',10,'LineWidth',2);
plot(p.sample(idx1,1),p.sample(idx1,2),'*b','MarkerSize',10,'LineWidth',2);

legend(sprintf('Class 0(%d)',length(idx0)),sprintf('Class 1(%d)',length(idx1)), 'Location','EastOutside');
end