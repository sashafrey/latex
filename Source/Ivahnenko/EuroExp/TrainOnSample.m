function [w] = TrainOnSample(task)
warning off
[w] = glmfit(task.objects(:,1:end-1), task.target - 1, 'binomial', 'link', 'logit');
warning on
w = [w(2:end); w(1)];
w = w ./ sqrt(w'* w);    
end