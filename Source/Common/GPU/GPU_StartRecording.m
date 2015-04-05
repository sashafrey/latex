function GPU_StartRecording(filename)
    % GPU_STARTRECORDING
    % Starts recording of all operations on GPU so that they can be
    % replayed later (for debugging purposes).
    %
    % Usage:
    % GPU_STARTRECORDING(filename), where
    %   filename  - file name of the file to store recorded blob
    Check(exist('filename', 'var'));

    GPU_Initialize;
    errco = calllib(GPU_LibName, 'startRecording', filename);
    GPU_CheckErrCode(errco);
end
