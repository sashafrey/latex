function allchildren = GetAllChildren(edges, a)
    task = QueueCreate();
    task = QueuePush(task, a);
    allchildren = VectorCreate();
    allchildren = VectorAdd(allchildren, a);
    while (~QueueIsEmpty(task))
        [task, alg] = QueuePop(task);
        algChildren = GetChildren(edges, alg);
        for child = algChildren
            if (child == a)
                continue;
            end
            if (~VectorContains(allchildren, child))
                allchildren = VectorAdd(allchildren, child);
                task = QueuePush(task, child);
            end            
        end
    end
    
    allchildren = VectorTrim(allchildren);
    allchildren = allchildren.Data';
end