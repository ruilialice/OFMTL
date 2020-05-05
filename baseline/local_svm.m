function [rmse_avg] = local_svm(Xtrain, Ytrain, Xtest, Ytest, lambda, opts)
%local_svm   for every task, train a svm model, and average the rmse over
%all tasks
%   input  
%        Xtrain  training data
%        Ytrain  label of training data
%        Xtest   testing data
%        Ytest   label of testing data
%        lambda  hyperparameter
%        opts
%        opts.max_sdca_iters   # of iteration
rmse = zeros(length(Xtrain), 1);
Yte_predict = cell(length(Xtrain), 1);
for i = 1:length(Xtrain)
    model = svm(Xtrain{i}, Ytrain{i}, lambda, opts);
    Yte_predict{i} = sign(Xtest{i} * model);
    rmse(i) = mean(Yte_predict{i} ~= Ytest{i});
end
rmse_avg = mean(rmse);
end

