function [bound, worstBound] = asymptoticUpperBound(algs, L, ell)
    nAlgs = algs.Count;

    errMatrix = zeros(nAlgs, L);
    for i = 1:nAlgs
        errMatrix(i, :) = algs.Data(i).errVect;
    end
    
    for i = nAlgs:-1:1
        if sum(errMatrix(i, :)) > ell
            errMatrix(i, :) = [];
        end
    end        
    
    [bound, worstBound] = asymptoticUpperBound_inner(errMatrix, ell, L);
end

function [ expectation , worst ] = asymptoticUpperBound_inner( A, l, L )

% A - is a matrix of errors
% l = L / 2 as usual

if 2*l ~= L
   error('l ~= L / 2');  
end

A = unique(A, 'rows');

%go
num_of_algo = size(A, 1);

%x = (log(num_of_algo) / l)^0.5;

%lambda = 1;
m_max = max(sum(A,2));
vect_of_classes = [];

for k = 0:m_max
    vect_of_classes = [vect_of_classes length(find(sum(A,2) == k))];
end


%for it = 1:number_of_grad_iter
%    value1 = 0;
%    value2 = 0;
%    for j = 1:m_max
%        vec = zeros(1, m_max + 1);
%        for l = j:m_max
%            vec(l + 1) = nchoosek(l, j);
%        end
%    val = ((vec * vect_of_classes') * (2^j * sqrt(pi) * nchoosek(-j + 2*l, -j + l) * gamma(1 - j + l)))/ ...
%    (gamma((1 - j)/2) * gamma(1/2 * (2 - j + 2*l)) * nchoosek(2*l, l));
%    value1 = value1 + j * val * ((tanh(x))^(j - 1)) * ((1 / cosh(x))^2);
%    value2 = value2 + val * (tanh(x))^(j);
%    end
%    value2 = value2 + num_of_algo;
%    current_grad = (x * l * (2 * l * sinh(x) + value1 / value2) -...
%        l * (log(value2)) + 2 * l * log(cosh(x)) ) / (x^2 * l^2);
%    
%    x = x - lambda * current_grad;
%    
%    Objective = 0;
%    for j = 0:m_max
%        vec = zeros(1, m_max + 1);
%        for l = j:m_max
%            vec(l + 1) = nchoosek(l, j);
%        end
%       Objective = Objective + ((vec * vect_of_classes') * (2^j * sqrt(pi) * nchoosek(-j + 2*l, -j + l) * gamma(1 - j + l)))/ ...
%     ( gamma((1 - j)/2) * gamma(1/2 * (2 - j + 2*l)) * nchoosek(2*l, l)) * (tanh(x))^j;
%    end
%
%    disp(Objective)
%    
%end

iter = 0.01:0.001:15;
z = [];
obj = inf;

for x = iter
    Objective = 0;
    for j = 0:2:m_max
        vec = zeros(1, m_max + 1);
        pre_calc =  log_factorial(2 * l - j) + log_factorial(j - 1) + log_factorial(l) -...
                    log_factorial(l - j / 2) - log_factorial(2 * l) - log_factorial(j / 2 - 1) - log_factorial(j);
        for r = j:m_max
            if j == 0 
                calc = 1;
            else
                calc = log_factorial(r) + pre_calc - log_factorial(r - j);
                
                calc = 2 * (-1)^(j/2) * exp(calc);
            end
        
            vec(r + 1) = calc;
        end
        %calc = (2^j * sqrt(pi) * nchoosek(-j + 2*l, -j + l) * gamma(1 - j + l))/ ...
        %( gamma((1 - j)/2) * gamma(1/2 * (2 - j + 2*l)) * nchoosek(2*l, l));
        
       Objective = Objective + (vec * vect_of_classes') * (tanh(x))^j;
    end
    
    Objective = (log(Objective) +  2 * l * log(cosh(x))) / (x * l);
    
    if Objective <= obj
        obj = Objective;
        z = [z x];
        %disp(Objective)
    else
       break; 
    end
end

expectation = obj;

%x = sdpvar(1, 1);
%Constraints = [x >= 0, x <= l * (log(num_of_algo) / l)^0.5];

%Objective = 0;
%for j = 0:m_max
%    vec = zeros(1, m_max + 1);
%    for l = j:m_max
%        vec(l + 1) = nchoosek(l, j);
%    end
%   Objective = Objective + ((vec * vect_of_classes') * (2^j * sqrt(pi) * nchoosek(-j + 2*l, -j + l) * gamma(1 - j + l)))/ ...
% ( gamma((1 - j)/2) * gamma(1/2 * (2 - j + 2*l)) * nchoosek(2*l, l)) * (tanh(x))^j;
%end

%Objective = (log(Objective) +  2 * l * log(cosh(x))) / (x * l);
%Objective = (((cosh(x))^(2 * l))*Objective)^(1/(x * l));
%options = sdpsettings('solver','fmincon', 'debug', 1);
%sol = solvesdp(Constraints, Objective); %, options);
%if sol.problem == 0
% x = double(x);
%else
% display('Hmm, something went wrong!');
% sol.info
% yalmiperror(sol.problem)
%end


%x = z(end);


%Objective = 0;

%for i = 1:num_of_algo
%   
%    m = sum(A(i,:));
%    value = 1;
%    
%    for j = 1:m
%       value = value + (((nchoosek(m, j)) * (2^j * sqrt(pi) * nchoosek(-j + 2*l, -j + l) * gamma(1 - j + l))/ ...
%     ( gamma((1 - j)/2) * gamma(1/2 * (2 - j + 2*l)) * nchoosek(2*l, l))));% * (tanh(x))^j;
%    end
%    Objective = Objective + value;
%end

%Objective = (log(Objective) +  l * x^2) / (x * l);

worst = 2 * (log(num_of_algo) / l)^0.5;
%disp([valuer obj worst])


end


