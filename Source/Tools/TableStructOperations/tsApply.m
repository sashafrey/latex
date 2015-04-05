function Target = tsApply(Source, Target)
    % ��������� ��������, ���������� � ��������� Source, � ���������
    % Target. ������� ��� �� ��������� ���� "����������" (�� ������ �����).
    % ������� ���������� �������� �� "Insert" (� ���������� "with replase")
    if (isempty(Source))
        return;
    end
    
    Check(tsIsValid(Source));
    Check(tsIsValie(Target));
    fields = fieldnames(Source);
    for i=1:length(fields)
        field = fields{i};
        Check(isfield(Target, field));
        if (isstruct(Source.(field)))
            Check(isstruct(Target.(field)), '������������� ���������');
            Target.(field) = tsAppend(Source.(field), Target.(field));
        else
            Target.(field) = Source.(field);
        end
    end
    Check(tsIsValid(Target));
end