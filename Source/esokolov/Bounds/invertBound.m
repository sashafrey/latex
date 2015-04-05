function invertedBound = invertBound(boundFunc, eta)
% boundFunc(eps) - ������� ������ ���������;
% ��������������, ��� ������� ��������� ������������ �� ������� [0, 1]

    precisionEps = 1e-3;
    left = 0;
    right = 1;
    
    while right - left > precisionEps
        mid = (left + right) / 2;
        currBound = boundFunc(mid);
        if currBound < eta
            right = mid;
        else
            left = mid;
        end
    end
    
    invertedBound = (left + right) / 2;
end