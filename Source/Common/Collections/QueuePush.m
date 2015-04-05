function [ queue ] = QueuePush( queue, value )
    if (isempty(queue.Data))
        queue.Data = value;
    else
        queue.Data(queue.Head, :) = value;
    end 
    
    queue.Head = queue.Head + 1;
    if (queue.Head > size(queue.Data, 1))
        queue.Head = 1;
    end
    
    if (queue.Head == queue.Tail)
        queue = Resize(queue);
    end
end

function queue = Resize(queue)
    count = size(queue.Data, 1);
    data = zeros(2 * count, size(queue.Data, 2));
    secondPartLength = count - queue.Tail + 1;

    data(1 : secondPartLength, :) = queue.Data(queue.Tail : count, :);
    data((secondPartLength + 1) : count, :) = queue.Data(1 : (queue.Head - 1), :);
    queue.Data = data;
    queue.Tail = 1;
    queue.Head = count + 1;    
end