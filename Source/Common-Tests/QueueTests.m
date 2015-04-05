function QueueTests
    % Tests for Queue
    queue = QueueCreate();
    assert(QueueCount(queue) == 0);
    for j=1:10
        for i=1:(5*j + 1)
            queue = QueuePush(queue, [i, i+1, 2*i]);
            assert(QueueCount(queue) == i);
        end

        for i=1:(5*j + 1)
            [queue, value] = QueuePop(queue);
            assert(all(value == [i, i+1, 2*i]));
            assert(QueueCount(queue) == (5*j + 1 - i));
        end
        
        assert(QueueIsEmpty(queue));
        assert(QueueCount(queue) == 0);
    end
end