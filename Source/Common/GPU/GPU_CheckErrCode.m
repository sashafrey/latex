function GPU_CheckErrCode(errco) 
    % GPU_CheckErrCode
    % Checks return code of internal GPU library. This method is for
    % internal usage only.

    if (errco < 0)
         ME = MException('GPU:failure', 'Some failure happened. Check LinearSampling.log for more details.');
          throw(ME);
    end
end