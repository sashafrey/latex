function GetGuidTests
    g1 = GetGUID;
    g2 = GetGUID;
    Check(g1 ~= g2, 'Guid must be unique.')
end
