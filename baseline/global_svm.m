function [rmse_avg] = global_svm(Xtrain, Ytrain, Xtest, Ytest, lambda, opts)
%global_svm  
%   input
%        Xtrain  training data
%        Ytrain  label of training data
%        Xtest   testing data
%        Ytest   label of testing data
%        lambda  hyperparameter
%        opts
%        opts.max_sdca_iters   # of iteration

% concatenate all training data together
Xtr_concat = cat(1, Xtrain{:});
Ytr_concat = cat(1, Ytrain{:});

% train step
model = svm(Xtr_concat, Ytr_concat, lambda, opts);

% test step
rmse = zeros(length(Xtrain), 1);

Yte_predict = cell(length(Xtrain), 1);

for i = 1:length(Xtrain)
    Yte_predict{i} = sign(Xtest{i} * model);
    rmse(i) = mean(Ytest{i} ~= Yte_predict{i});
end

rmse_avg = mean(rmse);
end

