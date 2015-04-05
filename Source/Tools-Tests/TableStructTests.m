function TableStructTests
    ts.x = [1; 2; 3];
    ts.y = [2; 3; 4];
    ts.z = {'a'; 'bb'; 'ccc'};
    ts.f = {@sin; @cos; @tan};
    
    Check(tsIsValid(ts), 'tsIsValid');
    Check(tsLength(ts) == 3, 'tsLenght');
    
    ts = tsConcat(ts, ts);
    Check(tsLength(ts) == 6, 'tsConcat');
    
    tsS = tsSort(ts, 'x');
    Check(issorted(tsS.x), 'tsSort - is sorted');
    Check(tsS.y(tsS.x == 2) == ts.y(ts.x == 2), 'tsSort - is consistent');
    
    tsG = tsGroup(ts, 'x');
    Check(sum(tsG.Count) == 6, 'tsGroup');
    
    tsX = tsSelect(ts, ts.x == 1);
    Check(tsLength(tsX) == 2, 'tsSelect');
    
    tsString = tsToString(ts);
    Check(~isempty(tsString), 'tsToString');
    
    tsR = tsRemove(ts, ts.x == 1);
    Check(tsLength(tsR) == 4, 'tsSelect');
end