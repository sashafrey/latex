function GPU_SetLogLevel(logLevel)
    % GPU_SetLogLevel
    % Sets minimal logging level.
    %
    % Usage: GPU_SetLogLevel(logLevel), where
    %   - logLevel must be one of the following:
    %       0 - to disable logging
    %       1 - for ERRORS (only)
    %       2 - for WARNING and higher
    %       3 - for INFO and higher
    %       4 - for DEBUG and higher
    errco = calllib(GPU_LibName, 'setLogLevel', logLevel);
    GPU_CheckErrCode(errco);
end