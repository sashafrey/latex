function Target = tsApply(Source, Target)
    % Сохраняет значения, записанные в структуре Source, в структуру
    % Target. Требует что бы структуры были "совместимы" (по списку полей).
    % Следует доработать операцию до "Insert" (с параметром "with replase")
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
            Check(isstruct(Target.(field)), 'Несовместимые структуры');
            Target.(field) = tsAppend(Source.(field), Target.(field));
        else
            Target.(field) = Source.(field);
        end
    end
    Check(tsIsValid(Target));
end