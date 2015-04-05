function GPU_ReplayRecording(filename)
    % GPU_ReplayRecording
    % Replayis the recording from given file.
    %
    % Usage:
    % GPU_REPLAYRECORDING(filename), where
    %   filename  - file name of the recorded blob to replay.
    Check(exist('filename', 'var'));

    GPU_Initialize;
    errco = calllib(GPU_LibName, 'replayRecording', filename);
    GPU_CheckErrCode(errco);
end
