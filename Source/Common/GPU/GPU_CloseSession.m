function GPU_CloseSession(sessionId)
    % GPU_CloseSession
    % Closes given session on GPU device.
    %
    % Usage: GPU_CloseSession(sessionId), where
    %   - sessionId is an identificator of session, produced by
    %   GPU_CreateSession() call.
    
    errco = calllib(GPU_LibName, 'closeSession', sessionId);
    GPU_CheckErrCode(errco);
end