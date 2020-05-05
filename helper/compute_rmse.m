function [ err ] = compute_rmse(X, Y, W, opts)
% Computes RMSE for MTL
% X: m-length cell of nxd features
% Y: m-length cell of nx1 labels
% W: dxm weight matrix
% opts:
%   opts.avg: compute avg (opts.avg = 1) or total (opts.avg = 0) rmse
%   opts.obj: 'R' for regression, 'C' for classification

%% compute predicted values
m = length(X);
Y_hat = cell(m,1);
for t=1:m
    Y_hat{t} = sign(X{t} * W(:, t));
end

%% compute errors
all_errs = zeros(m,1);
for t=1:m
    all_errs(t) = mean(Y{t} ~= Y_hat{t});
end
% compute mean error
err = mean(all_errs);

end

