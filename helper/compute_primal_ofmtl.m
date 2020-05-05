function [ primal_obj ] = compute_primal_ofmtl(X, Y, W, Omega, lambda)
% Inputs
%   Xtrain: input training data (m-length cell)
%   Ytrain: output training data (m-length cell)
%   W: current models (d x m)
%   Omega: precision matrix (m x m)
%   lambda: regularization parameter
% Output
%   primal objective

% compute primal
total_loss = 0;
n = length(X);

preds = Y{n}.*(X{n}*W(:, n));
total_loss = total_loss + mean(max(0.0, 1.0 - preds));

primal_obj = total_loss + lambda / 2 * trace(W * Omega * W');

end
