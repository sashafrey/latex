function [err] = CalcErrorOnSample(task, w)
err = mean((2 * task.target-3) .* (task.objects * w) <= 0);
end