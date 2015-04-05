function C50Clean(task)
    delete(sprintf('C:\\temp\\%s.*', task.filename));
end