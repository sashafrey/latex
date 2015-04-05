function [ fact ] = log_factorial( n )
%log factorial 
%   

% fact = 0;
% 
% for i = 1:n
%     fact = fact + log(i);
% end

fact = sum(log(1:n));

end

