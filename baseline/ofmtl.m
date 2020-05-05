function [W, Sigma, rmse, primal_obj] = ofmtl(Xtrain, Ytrain, Xtest, Ytest, Sigma, W, lambda, opts)
% Currently, there're m-1 tasks in the system, and here comes the m task
% Inputs
%   Xtrain: input training data (1*m cell, every cell is n_i * d matrix)
%   Ytrain: output training data (1*m cell, every cell is n_i * 1 matrix)
%   Xtest: input test data
%   Ytest: output test data
%   Sigma: covariance matrix of W ((m-1) * (m-1) matrix)
%   W: model parameter (d * (m-1) matrix)
%   lambda: regularization parameter
%   opts: optimal arguments
% Output
%   W: model parameter (d * m matrix)
%   Sigma: covariance matrix of W (m * m matrix)
%   rmse: average rmse across tasks in each outer iteration (opts.mocha_outer_iters * 1 matrix)

%% intialize variables
m = size(Xtrain, 2); % # of tasks in the current system
nm = length(Ytrain{m}); % # of data on m_th task
d = size(Xtrain{1}, 2);
alpha = zeros(nm, 1); % alpha of every data on task m
rmse = zeros(opts.mocha_outer_iters, 1);

%% intialize Sigma and Omega
empty = zeros(m-1, 1);
Sigma = [Sigma empty];
empty = zeros(m, 1);
Sigma = [Sigma; empty'];
Sigma = 1.0 * (m-1) / m * Sigma;
Sigma(m, m) = 1.0 / m;
Omega = inv(Sigma);

Wm = zeros(d, 1);
W = [W Wm];

primal_obj = zeros(opts.mocha_outer_iters, 1);
for h = 1:opts.mocha_outer_iters
    %% compute rmse, dual and primal in online MTL
    curr_err = compute_rmse(Xtest, Ytest, W, opts);
    rmse(h) = curr_err;
    primal_obj(h) = compute_primal_ofmtl(Xtrain, Ytrain, W, Omega, lambda);
    if(h>1 && abs(primal_obj(h)-primal_obj(h-1))<1e-5)
        primal_obj(h+1:opts.mocha_outer_iters) = primal_obj(h);
        rmse(h+1:opts.mocha_outer_iters) = rmse(h);
        break;
    end
    
    for hh = 1:opts.mocha_inner_iters
        rng(1000*hh);
        deltaWm = zeros(d, 1);
        mperm = randperm(nm);
        curr_omg = Omega(m, m);
        
        if(opts.sys_het)
            sys_iters = (opts.top - opts.bottom) .* rand(m,1) + opts.bottom;
            local_iters = nm * sys_iters;
        else
            local_iters = nm * opts.mocha_sdca_frac;
        end
        
        % run SDCA locally
        for s=1:local_iters
            idx = mperm(mod(s, nm) + 1);
            alpha_idx_old = alpha(idx);
            curr_y = Ytrain{m}(idx);
            curr_x = Xtrain{m}(idx, :);
            
            % compute update
            update = curr_y * curr_x * (Wm + deltaWm);
            grad = lambda * nm * (1.0 - update) * curr_omg / (curr_x * curr_x') + (alpha_idx_old * curr_y);
            alpha(idx) = curr_y * max(0.0, min(1.0, grad));
            deltaWm = deltaWm + 1.0 * (alpha(idx) - alpha_idx_old) * curr_x' / (nm * lambda * curr_omg);           
        end
        
        %% update Wm
        A = zeros(d, 1);
        B = zeros(d, 1);
        for tt = 1:m-1
            A = A + lambda * Omega(m, tt) * W(:, tt);
        end
        for tt = 1:nm
            B = B + 1.0 * alpha(tt) * (Xtrain{m}(tt, :)')/nm;
        end
        Wm = ((-1.0 * A) + B) / (lambda * Omega(m, m));
    end
    
    W(:, m) = Wm;
    
    %% make sure eigenvalues are positive
    AA = W' * W;
    if(any(eig(AA)) < 0)
        [V, Dmat] = eig(AA);
        dm = diag(Dmat);
        dm(dm <= 1e-7) = 1e-7;
        D_c = diag(dm);
        AA = V*D_c*V';
    end
    
    %% update Omega, Sigma
    sqm = sqrtm(AA);
    Sigma = sqm / trace(sqm);
    Omega = inv(Sigma);
                        
end

end

