function [model] = svm(X, Y, lambda, opts)
%svm  
%   input 
%        X   training data
%        Y   training label
%        lambda  hyperparameter
%        opts

%   output
%        model  svm model parameter
[N, dim] = size(X);
model = zeros(dim, 1);
alpha = zeros(N, 1);
primal_old = 0;
opts.tol = 1e-5;

for i = 1:opts.max_sdca_iters
    for n = 1:N
        alpha_old = alpha(n);
        x = X(n, :);
        y = Y(n);
        grad = lambda * N * (1.0-(y*x*model)) / (x * x') + (alpha_old * y);
        
        alpha(i) = y * max(0, min(1.0, grad));
        model = model + (alpha(i) - alpha_old) * x' * (1.0/(lambda * n));
    end
    
    predict = Y .*(X * model);
    primal_new = mean(max(0.0, 1.0 - predict)) + (lambda / 2.0) * (model' * model);
    if(abs(primal_old - primal_new) < opts.tol)
        break;
    end
    primal_old = primal_new;
end
end

