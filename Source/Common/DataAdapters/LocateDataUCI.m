function folder = LocateDataUCI()
    curpath = '.';
    for i = 1:5
        located = false;
        folders = dir(curpath)';
        for tmpfolder = folders
            if (strcmp(tmpfolder.name, 'DataUCI'))
                folder = fullfile(curpath, '\', tmpfolder.name);
                located = true;
                break;
            end
        end

        if (located)
            break;
        end

        curpath = ['..\', curpath];
    end

    if (~located)
        warning('Unable to locate DataUCI folder, tasks haven''t been loaded');
        folder = [];
        return;
    end
end