function plotQ(names, varargin)
markers = ['v','o','x','s','+'];

figure;
hold on;
for j=1:nargin-1
    [x,y] = pl(varargin{j});
    plot(x,y,sprintf('-%sk',markers(j)));
end;
legend(names');
%plot(p.Q(:,1),ones(size(p.Q,1),1),'r*');
%[x,y] = pl(p.Q);
%plot(x,y,'-r');
%[x,y] = pl(p.realFixQ);
%plot(x,y,'-g');
%[x,y] = pl(p.fastFixQ);
%plot(x,y,'-b');
%if (p.mode=='c')
    %[x,y] = pl(p.voronQ_real);
    %[x,y] = pl(p.voronQ);
    %plot(x,y,'-c');
    %legend('voronQ');
%end;
%legend('eps','Q','realFixedQ','fastFixedQ','voronQ');
%legend('Q','realFixedQ','fastFixedQ','voronQ');
hold off;

function [x,y] = pl(q)
x = q(:,1);
y = zeros(size(q,1),1);
for i=1:length(y)
    y(i) = sum(q(i:end,2));
end;
end

end
