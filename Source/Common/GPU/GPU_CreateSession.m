function sessionId = GPU_CreateSession(X, target, R, deviceId)
    % GPU_CREATESESSION 
    % Creates a session in GPU device.
    %
    % Usage:
    % sessionId = GPU_CREATESESSION(X, target, R, deviceId), where
    %   - X is a matrix of size [nItems x nFeatures], also known as feature
    %   matrix;
    %   - target is a vector of size [nItems] that describes labels of
    %   target classes. Labels must be either 0 or 1;
    %   - R is a matrix of size [nRays x nFeatures, that gives a set of
    %   pre-defined directions used to search for neighbours of
    %   classifiers;
    %   - deviceId defines that id of GPU device. Optional, default = 0;
    %  - sessionId is an identificator of newly created session.
    
    if (~exist('deviceId', 'var'))
        deviceId = 0;
    end
    
    GPU_Initialize;
    nItems = size(X, 1);
    nFeatures = size(X, 2);
    nRays = size(R, 1);
    Check(size(R, 2) == nFeatures);
    
    [X, target, R] = Prepare(X, target, R);
    
    Check(length(target) == nItems);
    Check(all(ismember(target, [0, 1])));

    sessionId = calllib(GPU_LibName, 'createSession', X, target, R, int32(nItems), int32(nFeatures), int32(nRays), int32(deviceId), int32(-1));
    GPU_CheckErrCode(sessionId);
end

function [X, target, R] = Prepare(X, target, R)
    if (~isa(X, 'single'))
        X = single(X);
    end
    
    if (isa(target, 'int32'))
        target = int32(target);
    end
    
    if (~isa(R, 'single'))
        R = single(R);
    end
end