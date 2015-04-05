function [ result ] = TestsLauncherInternal( dirName, debug )
%TestsLauncherInternal: function, with argument for recursive call
%organization
%
%   Input:
%   dirName: directoryName to scan it for unitTest files
%   dirOnly: bool flag. If "1" we just process all directories,
%   without lauching files
%   offset: for make offset in console-output (tree-view visualization)
%   
%   Output:
%   result: structure, has two fields - bool flag success, number of failed
%   tests
%   
%   Example:
%   TestsLauncherInternal('tests', 1, 0) - runs all tests in
%   subdirectories, will skip all directories
%   tests
result.passed = 0;
result.failed = 0;

addpath(dirName);
a = dir(dirName);
for file = [1:length(a)]
    curFile = a(file).name;
    if (curFile(1) == '.') 
        continue; 
    end;
    
    if (a(file).isdir == 1)
        nextFile = strcat(dirName, '\', curFile);
        tresult = TestsLauncherInternal( nextFile, 0, offset + 4);
        result.passed = result.passed + tresult.passed;
        result.failed = result.failed + tresult.failed;
    end
    if (strfind(curFile, '.m') == length(curFile) - 1)
        % we found executable .m-file. Run it!
        fprintf('%s\\%s... ', dirName, curFile(1:length(curFile) - 2));

        command = strcat(curFile(1:length(curFile) - 2), '()');
        if (debug)
            eval(command);
            result.passed = result.passed + 1;
            fprintf(' OK\n');
        else
            try
                eval(command);
                result.passed = result.passed + 1;
                fprintf(' OK.\n');
            catch ex
                result.failed = result.failed + 1;
                cprintf([1, 0, 0], sprintf('%s\\\\%s... Failed\n', dirName, curFile(1:length(curFile) - 2)));
            end
        end
    end
end
return;