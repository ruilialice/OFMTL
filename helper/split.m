function [Xtrain, Ytrain, Xtest, Ytest] = split(X, Y, training_percent)
%split split X and Y according to the training_percent to Xtrain, Ytrain, Xtest, Ytest
%% input
%   X data
%   Y label
%   training_percent training_percent
%% output
%   Xtrain training data
%   Ytrain label of training data
%   Xtest  testing data
%   Ytest  label of testing data
N = size(X, 2);
Xtrain = cell(N, 1);
Ytrain = cell(N, 1);
Xtest = cell(N, 1);
Ytest = cell(N, 1);
for n=1:N
    data_size = length(Y{n});
    tSelIdx = randperm(data_size);
    train_size = floor(data_size * training_percent);
    Xtrain{n} = X{n}(tSelIdx(1:train_size), :);
    Ytrain{n} = Y{n}(tSelIdx(1:train_size), :);
    Xtest{n} = X{n}(tSelIdx(train_size+1:end), :);
    Ytest{n} = Y{n}(tSelIdx(train_size+1:end), :);
end

end

