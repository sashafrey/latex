function JavaHashtableTest
    % More functions: http://docs.oracle.com/javase/1.4.2/docs/api/java/util/Hashtable.html
    hashtable = javaObject('java.util.Hashtable');
    hashtable.put(1, 105);
    hashtable.put(5, 232);
    hashtable.put(3, 153);
    
    Check(hashtable.containsKey(1));
    Check(hashtable.get(1), 105);    
    Check(hashtable.size() == 3);
    
    hashtable.clear();
    Check(hashtable.size() == 0);
    
    % You can put arrays and cell-arrays into hashtable:
    hashtable.put(1, [1 2 3]);
    hashtable.put(2, [2 3 4]');
    hashtable.put(3, {'a' 'bc' 'xyz'});
    
    % WARNING: both arrays are stored in a horizontal way:
    Check(all(hashtable.get(1) == [1 2 3]'));
    Check(all(hashtable.get(2) == [2 3 4]'));
    
    % WARNING: cell array is converted to some java array of objects. Now
    % you should you () instead of {}.
    ca = hashtable.get(3);
    Check(strcmp(ca(1), 'a'));
    Check(strcmp(ca(2), 'bc'));
    Check(strcmp(ca(3), 'xyz'));
end