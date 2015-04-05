function [ ] = CVT( debug )
    % CVT - run check-in validation tests. 
    %
    % These tests are required to run before each check-in (e.g. before 
    % each commit to public repository like SVM).
    % 
    % Syntax:
    %   CVT - just run it without arguments. That is it. Simple!
    %   CVT(true) - run with 'true' argument to stop execution after first error.
    
    if (nargin == 0)
        debug = 0;
    end
    
    result.passed = 0;
    result.failed = 0;
    result = Append(result, TestsLauncherInternal('Tools-Tests', debug));
    result = Append(result, TestsLauncherInternal('Common-Tests', debug));
    
    if (result.failed == 0)
        sessionMessage = sprintf('%s %d out of %d tests passed\n', 'CVT session succeeded, ', result.passed, result.passed + result.failed);
        cprintf([0,0.7,0.2], sessionMessage);
    else 
        sessionMessage = sprintf('%s %d out of %d tests failed\n', 'CVT session failed, ', result.failed, result.passed + result.failed);
        cprintf([1, 0, 0], sessionMessage);
    end;
end

function result = Append(result, result2)
    result.passed = result.passed + result2.passed;
    result.failed = result.failed + result2.failed;
end