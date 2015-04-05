%Генерация простой, хорошо разделимой выборки cо случайным шумом
function [sample, sampleClasses] = GenerateSimpleSampleWithRandomNoise(L, dim, noiseObjects)
    %L -длина выборки
    %dim - размерность пространства объектов
    if nargin < 3
        noiseObjects = 1
    end
    if nargin < 2
        dim = 2;
    end

    [sample, sampleClasses] = GenerateSimpleSample(L, dim);
    rp = randperm(L);
    sampleClasses(rp(1:noiseObjects)) = -sampleClasses(rp(1:noiseObjects));
end