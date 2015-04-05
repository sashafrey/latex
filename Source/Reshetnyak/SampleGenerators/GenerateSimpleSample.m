%Генерация простой, хорошо разделимой выборки
function [sample, sampleClasses] = GenerateSimpleSample(L, dim)
    %L -длина выборки
    %dim - размерность пространства объектов
    if nargin < 2
        dim = 2;
    end

    sample = [ mvnrnd( repmat(0, L/2, dim), 3 * eye(dim)) ; mvnrnd( repmat(10, L/2, dim), 3 * eye(dim))];
    sampleClasses = [ones(1, L/2) -ones(1, L/2)];
end