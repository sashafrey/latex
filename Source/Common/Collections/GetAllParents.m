function allparents = GetAllParents(edges, a)
    task = QueueCreate();
    task = QueuePush(task, a);
    allparents = VectorCreate();    
    allparents = VectorAdd(allparents, a);
    while (~QueueIsEmpty(task))
        [task, alg] = QueuePop(task);
        algParents = GetParents(edges, alg);
        for parent = algParents
            if (parent == a)
                continue;
            end
            if (~VectorContains(allparents, parent))
                allparents = VectorAdd(allparents, parent);
                task = QueuePush(task, parent);
            end            
        end
    end
    
    allparents = VectorTrim(allparents);
    allparents = allparents.Data';
end