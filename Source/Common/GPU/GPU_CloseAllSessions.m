function GPU_CloseAllSessions
    % GPU_CloseAllSessions
    % Closes all active sessions on GPU device.
    
    errco = calllib(GPU_LibName, 'closeAllSessions');
    GPU_CheckErrCode(errco);
end