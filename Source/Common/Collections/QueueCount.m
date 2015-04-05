function count = QueueCount( queue )
    if (queue.Head >= queue.Tail)
        count = queue.Head - queue.Tail;
    else
        count = size(queue.Data, 1) + queue.Head - queue.Tail;
    end    
end