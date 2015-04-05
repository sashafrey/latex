function [sample, sampleClasses] = GenerateCloseClasses(L, dim)
    %L -длина выборки
    %dim - размерность пространства объектов
    if nargin < 2
        dim = 2
    end

    sample = [ mvnrnd( repmat(0, L/2, dim), eye(dim)) ; mvnrnd( repmat(3.5*dim^(-0.5),L/2, dim), eye(dim))];   
    sampleClasses = [ones(1, L/2) -ones(1, L/2)];
end