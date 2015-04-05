function GPU_StopRecording()
    % GPU_STOPRECORDING
    % Stops ongoing recording.
    
    errco = calllib(GPU_LibName, 'stopRecording');
    GPU_CheckErrCode(errco);
end
