function [predictions, phi, w, phi_full] = rvmpredict(task, model)
    Xtest = task.objects;
    Xtest(isnan(Xtest)) = 0;
    phi	= SB1_KernelFunction(Xtest,model.X(model.used,:), model.kernel,model.width);
    phi_full = SB1_KernelFunction(Xtest,model.X, model.kernel,model.width);
    y_rvm	= phi * model.weights + model.bias;
    
    predictions = NaN(task.nItems, 1);
    predictions(y_rvm > 0) = 2;
    predictions(y_rvm <= 0) = 1;
    
    w = model.weights;
    if (model.bias ~= 0)
        phi(:, end+1) = ones(size(phi, 1), 1);
        phi_full(:, end+1) = ones(size(phi_full, 1), 1);
        w = [w; model.bias];
    end
end