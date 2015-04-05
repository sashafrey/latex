function GUID = GetGUID()
    guid = mexCreateGUID;
    guid(guid=='{' | guid=='}' | guid=='-') = [];
    GUID = hex2dec(guid);    % todo: заботать uint64-guid.
end