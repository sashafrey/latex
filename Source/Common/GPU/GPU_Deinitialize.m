function GPU_Deinitialize()
    % GPU_Deinitialize
    % Unloads the GPU library from main memory.
    if (libisloaded(GPU_LibName))
        GPU_CloseAllSessions();
        unloadlibrary(GPU_LibName);
    end
end