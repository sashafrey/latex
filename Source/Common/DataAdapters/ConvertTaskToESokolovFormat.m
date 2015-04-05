function [ X, Y ] = ConvertTaskToESokolovFormat( task )
    Check(task.nClasses == 2);
	X = task.objects;
    Y = 2 * (task.target - 1.5);
    %X = [X, ones(length(Y), 1)];
end

