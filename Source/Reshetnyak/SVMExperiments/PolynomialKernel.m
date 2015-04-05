function [res] = PolynomialKernel(x, y)
    kernelDegree = 4;
    %kernelDegree - global variable
    res = (1 + x .* y)^kernelDegree;
end