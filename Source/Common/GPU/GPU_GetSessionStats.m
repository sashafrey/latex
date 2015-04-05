function [nItems, nFeatures, nRays, deviceId] = GPU_GetSessionStats(sessionId)
    % GPU_GetSessionStats
    % Retrieves statiscits of the given GPU session.
    %
    % Usage: 
    % [nItems, nFeatures, nRays, deviceId] = GPU_GetSessionStats(sessionId), 
    % where
    %   - sessionId is an identificator of session, produced by
    %   GPU_CreateSession() call,
    %   - nItems gives the number of items (objects) in the task,
    %   - nFeatures gives the number of features in the task,
    %   - nRays gives the number of predefined search directions,
    %   - deviceId gives the id of GPU device that session belongs to.
    
    nItems = int32(0);
    nFeatures = int32(0);
    nRays = int32(0);
    deviceId = int32(0);
    
    [errco, nItems, nFeatures, nRays, deviceId] = calllib(GPU_LibName, 'getSessionStats', sessionId, nItems, nFeatures, nRays, deviceId);
    GPU_CheckErrCode(errco);
end