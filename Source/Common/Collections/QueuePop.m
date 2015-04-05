function [ queue, value ] = QueuePop( queue )
    if (QueueIsEmpty(queue)) 
        throw(MException('QueueUnderflow', 'QueueUnderflow'));
    end
    
    value = queue.Data(queue.Tail, :);
    queue.Tail = queue.Tail + 1;
    if (queue.Tail > size(queue.Data, 1))
        queue.Tail = 1;
    end
end
