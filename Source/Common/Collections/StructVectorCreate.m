function [ vector ] = StructVectorCreate(sampleValue)
    % —труктура, аналогична€ Vector, но дл€ хранени€ структур.
    % »з-за особенностей матлаба при инициализации уже нужно знать тип
    % хранимых структур, поэтому нужно передавать sampleValue ---
    % структуру, имеющую требуемый тип.
    
    vector.Data = sampleValue;
    vector.Data(1) = [];
    vector.Count = 0;
    
end

