%��������� �������, ������ ���������� ������� c� ��������� �����
function [sample, sampleClasses] = GenerateSimpleSampleWithRandomNoise(L, dim, noiseObjects)
    %L -����� �������
    %dim - ����������� ������������ ��������
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