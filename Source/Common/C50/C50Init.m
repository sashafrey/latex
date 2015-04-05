function C50Init()
    iRep = 10;
    while(iRep > 0)
        iRep = iRep - 1;
        ok = true;
        try
            ok = true;
            if (~exist('C:\\temp\\C50Tune.exe', 'file'))
                ok = false;
                copyfile('Common\\C50\\C50Tune.exe', 'C:\\temp\\C50Tune.exe');
            end

            if (~exist('C:\\temp\\C50Calc.exe', 'file'))
                ok = false;
                copyfile('Common\\C50\\C50Calc.exe', 'C:\\temp\\C50Calc.exe');
            end
        catch e
            fprintf('Error %e during C50 initialization', e);
        end
        
        if (ok)
            break;
        end
    end
end