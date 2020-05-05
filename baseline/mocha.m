function [rmse, alpha, W, Sigma, primal_objs] = mocha(Xtrain, Ytrain, Xtest, Ytest, lambda, opts)
% Mocha Method
% Inputs
%   Xtrain: input training data
%   Ytrain: output training data
%   Xtest: input test data
%   Ytest: output test data
%   lambda: regularization parameter
%   opts: optional arguments
% Output
%   Average RMSE across tasks, 
%   other parameters which may be useful for ofmtl, such as alpha, W and Sigma

%% intialize variables
fprintf('Running MOCHA\n');
m = length(Xtrain); % # of tasks
d = size(Xtrain{1}, 2); % # of features
W = zeros(d, m); 
alpha = cell(m,0);
Sigma = eye(m) * (1/m);
Omega = inv(Sigma);

totaln = 0; n = zeros(m, 1);
for t = 1:m
    n(t) = length(Ytrain{t});
    totaln = totaln + n(t);
    alpha{t} = zeros(n(t), 1);
end

%% intialize counters
rho = 1.0;
rmse = zeros(opts.mocha_outer_iters, 1);
primal_objs = zeros(opts.mocha_outer_iters, 1);


for h = 1:opts.mocha_outer_iters
    curr_err = compute_rmse(Xtest, Ytest, W, opts);
    rmse(h) = curr_err;
    primal_objs(h) = compute_primal(Xtrain, Ytrain, W, Omega, lambda);
    
    if(h>1 && abs(rmse(h-1)-rmse(h))<1e-5 && abs(primal_objs(h)-primal_objs(h-1))<1e-5)
        rmse(h+1 : opts.mocha_outer_iters) = rmse(h);
        break;
    else
        
        % update W
        for hh = 1:opts.mocha_inner_iters
            rng(hh*1000);
            if(opts.sys_het)
                sys_iters = (opts.top - opts.bottom) .* rand(m,1) + opts.bottom;
            end
            
            % loop over tasks (in parallel)
            deltaW = zeros(d, m);
            deltaB = zeros(d, m);
            for t = 1:m
                tperm = randperm(n(t));
                alpha_t = alpha{t};
                curr_sig = Sigma(t,t);
                if(opts.sys_het)
                    local_iters = n(t) * sys_iters(t);
                else
                    local_iters = n(t) * opts.mocha_sdca_frac;
                end
                
                % run SDCA locally
                for s=1:local_iters
                    % select random coordinate
                    idx = tperm(mod(s, n(t)) + 1);
                    alpha_old = alpha_t(idx);
                    curr_y = Ytrain{t}(idx);
                    curr_x = Xtrain{t}(idx, :);
                    
                    % compute update
                    update = (curr_y * curr_x * (W(:,t) + rho * deltaW(:, t)));
                    grad = lambda * n(t) * (1.0 - update) / (curr_sig * rho * (curr_x * curr_x')) + (alpha_old * curr_y);
                    alpha_t(idx) = curr_y * max(0.0, min(1.0, grad));
                    deltaW(:, t) = deltaW(:, t) + Sigma(t, t) * (alpha_t(idx) - alpha_old) * curr_x' / (lambda * n(t));
                    deltaB(:, t) = deltaB(:, t) + (alpha_t(idx) - alpha_old) * curr_x' / n(t);
                    alpha{t} = alpha_t;
                end
            end
            
            % combine updates globally
            for t = 1:m
                for tt = 1:m
                    W(:, t) = W(:, t) + deltaB(:, tt) * Sigma(t, tt) * (1.0 / lambda);
                end
            end
        end
        
        %% make sure eigenvalues are positive
        A = W'*W;
        if(any(eig(A) < 0))
            [V,Dmat] = eig(A);
            dm= diag(Dmat);
            dm(dm <= 1e-7) = 1e-7;
            D_c = diag(dm);
            A = V*D_c*V';
        end
        
        %% update Omega, Sigma
        sqm = sqrtm(A);
        Sigma = sqm / trace(sqm);
        Omega = inv(Sigma);
        rho = max(sum(abs(Sigma),2)./ diag(Sigma));
    end
end

end
