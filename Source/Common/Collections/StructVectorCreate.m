function [ vector ] = StructVectorCreate(sampleValue)
    % ���������, ����������� Vector, �� ��� �������� ��������.
    % ��-�� ������������ ������� ��� ������������� ��� ����� ����� ���
    % �������� ��������, ������� ����� ���������� sampleValue ---
    % ���������, ������� ��������� ���.
    
    vector.Data = sampleValue;
    vector.Data(1) = [];
    vector.Count = 0;
    
end

