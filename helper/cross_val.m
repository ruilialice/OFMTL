function [best_lambda] = cross_val(X, Y, baseline, opts, lambda_range, fold)
% cross_val  choosing the right lambda
%   input 
%        Xtrain    training data
%        Ytrain    label of training data
%        baseline  can be chosen from global_svm, local_svm, cocoa, mbsgd, mbsdca, mocha, ofmtl
%        opts      some parameters about optimization
%        lambda_range  input lambda_range
%        fold      # of cross validation fold    
%   output
%        best_lambda

func = str2func(baseline);

% # of tasks
N = length(Y);

% record the accuracy under every lambda
rmse_record = zeros(length(lambda_range), 1);
fprintf('Starting cross validation\n')
for idx_fold = 1:fold
    fprintf('fold %f   \n', idx_fold)
    Xtrain = cell(N, 1);
    Ytrain = cell(N, 1);
    Xtest = cell(N, 1);
    Ytest = cell(N, 1);
    
    for i = 1:N
        % # of data for task i
        num_data = length(Y{i});
        testidx = idx_fold:fold:num_data;
        trainidx = setdiff(1:num_data, testidx);
        Xtrain{i} = X{i}(trainidx, :);
        Ytrain{i} = Y{i}(trainidx, :);
        Xtest{i} = X{i}(testidx, :);
        Ytest{i} = Y{i}(testidx, :);        
    end
    
    for i = 1:length(lambda_range)
        if (strcmp(baseline,'global_svm') || strcmp(baseline,'local_svm'))
            rmse = func(Xtrain, Ytrain, Xtest, Ytest, lambda_range(i), opts);
        elseif (strcmp(baseline,'cocoa'))
            [rmse, ~, primal_objs] = func(Xtrain, Ytrain, Xtest, Ytest, lambda_range(i), opts);
        elseif (strcmp(baseline,'mocha'))
            [rmse, ~, ~, ~, primal_objs] = func(Xtrain, Ytrain, Xtest, Ytest, lambda_range(i), opts);
        else
            [rmse, primal_objs] = func(Xtrain, Ytrain, Xtest, Ytest, lambda_range(i), opts);
        end
        rmse_record(i) = rmse_record(i) + rmse(end);
    end
end
fprintf('Ending cross validation\n')
rmse_record = rmse_record ./fold;

[best_rmse, index] = min(rmse_record);
best_lambda = lambda_range(index);

end

