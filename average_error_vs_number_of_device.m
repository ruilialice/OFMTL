%% driver for average prediction error vs # of mobile devices

%% load dataset
datafolder = 'data/';
dataset = 'EAT_VOC';
load([datafolder dataset]);

%% set parameters
addpath('helper/'); addpath('baseline/');
ntrials = 5;
n_0 = 1; % # of initial tasks in the system
thre = 1.5; % threshold to perform global update
n_now = n_0; % # of tasks in the latest global update
N = size(X_ori, 2); % # of tasks in the dataset
training_percent = 0.75; % training percentage
opts.sys_het = 0;

lambda_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]; % set the hyperparameter search space

%% initializtion
err_local = cell(ntrials, 1);
err_global = cell(ntrials, 1);
err_mtl = cell(ntrials, 1);
err_ofmtl = cell(ntrials, 1);

time_local = cell(ntrials, 1);
time_global = cell(ntrials, 1);
time_mtl = cell(ntrials, 1);
time_ofmtl = cell(ntrials, 1);

for trial = 1:ntrials
    % permute the data
    rng(trial);
    tSelIdx = randperm(N);
    X_total = X_ori(tSelIdx);
    Y_total = Y_ori(tSelIdx);
    X = X_total(1:n_0); % X is the data currently in the system
    Y = Y_total(1:n_0);
    
    % separate the training and testing data
    [Xtrain, Ytrain, Xtest, Ytest] = split(X, Y, training_percent);
    
    % initialization
    err_local{trial} = zeros(N-n_0+1, 1);
    err_global{trial} = zeros(N-n_0+1, 1);
    err_mtl{trial} = zeros(N-n_0+1, 1);
    err_ofmtl{trial} = zeros(N-n_0+1, 1);
    
    time_local{trial} = zeros(N-n_0+1, 1);
    time_global{trial} = zeros(N-n_0+1, 1);
    time_mtl{trial} = zeros(N-n_0+1, 1);
    time_ofmtl{trial} = zeros(N-n_0+1, 1);
    
    % global model
    opts.max_sdca_iters = 500;
    tic
    global_lambda = cross_val(Xtrain, Ytrain, 'global_svm', opts, lambda_range, 5);
    err_global{trial}(1) = global_svm(Xtrain, Ytrain, Xtest, Ytest, global_lambda, opts);
    temp_global = toc
    time_global{trial}(1) = temp_global;
    
    % local model
    opts.max_sdca_iters = 500;
    tic
    local_lambda = cross_val(Xtrain, Ytrain, 'local_svm', opts, lambda_range, 5);
    err_local{trial}(1) = local_svm(Xtrain, Ytrain, Xtest, Ytest, local_lambda, opts);
    temp_local = toc
    time_local{trial}(1) = temp_local;
    
    %% MTL model (mocha)
    opts.mocha_outer_iters = 10;
    opts.mocha_inner_iters = 100;
    opts.mocha_sdca_frac = 0.5;
    opts.w_update = 0; % do a full run, not just one w-update
    opts.sys_het = 0; % not messing with systems heterogeneity
    tic
    mocha_lambda = cross_val(Xtrain, Ytrain, 'mocha', opts, lambda_range, 5); % determine via 5-fold cross val
    [rmse_mocha_reg, alpha, W, Sigma, ~] = mocha(Xtrain, Ytrain, Xtest, Ytest, mocha_lambda, opts);
    err_mtl{trial}(1) = rmse_mocha_reg(end);
    temp_mtl = toc
    time_mtl{trial}(1) = temp_mtl;
    % update lambda for ofmtl
    ofmtl_lambda = mocha_lambda;
    temp_ofmtl = temp_mtl;
    time_ofmtl{trial}(1) = temp_ofmtl;
    err_ofmtl{trial}(1) = rmse_mocha_reg(end);
    i_new = 0; % number of new tasks adding into the system after the latest global update
    
    n_now = n_0;

    for i=1:N-n_0
    %for i=1:m_total-m_0
        n = n_0 + i;
        i_new = n - n_now; % number of new tasks adding into the system after the latest global update
        rng(i*trial*100)
        [Xtemp, Ytemp, Xtest_temp, Ytest_temp] = split(X_total(n), Y_total(n), training_percent);
        Xtrain(n) = Xtemp;
        Ytrain(n) = Ytemp;
        Xtest(n) = Xtest_temp;
        Ytest(n) = Ytest_temp;
        
        %% global model
        opts.type = 'global';
        tic
        global_lambda = cross_val(Xtrain, Ytrain, 'global_svm', opts, lambda_range, 5);
        err_global{trial}(i+1) = global_svm(Xtrain, Ytrain, Xtest, Ytest, global_lambda, opts);
        temp = toc
        temp_global = temp_global + temp;
        time_global{trial}(i+1) = temp_global;
        
        %% local model
        opts.type = 'local';
        tic
        local_lambda = cross_val(Xtrain, Ytrain, 'local_svm', opts, lambda_range, 5);
        err_local{trial}(i+1) = local_svm(Xtrain, Ytrain, Xtest, Ytest, local_lambda, opts);
        temp = toc
        temp_local = temp_local + temp;
        time_local{trial}(i+1) = temp_local;
        
        %% MTL model
        tic
        mocha_lambda = cross_val(Xtrain, Ytrain, 'mocha', opts, lambda_range, 5);
        [rmse_mocha_reg, alpha, W_ori, Sigma_ori, ~] = mocha(Xtrain, Ytrain, Xtest, Ytest, mocha_lambda, opts);
        err_mtl{trial}(i+1) = rmse_mocha_reg(end);
        temp = toc
        temp_mtl = temp_mtl + temp;
        time_mtl{trial}(i+1) = temp_mtl;
        
        %% online MTL model
        if 1.0*i_new/n_now >= thre
            n_now = n_now + i_new;
            %% global update to choose the best lambda
            Sigma = Sigma_ori;
            W = W_ori;
            rmse = rmse_mocha_reg;
            err_ofmtl{trial}(i+1) = rmse_mocha_reg(end);
            temp_ofmtl = temp_ofmtl + temp;
            time_ofmtl{trial}(i+1) = temp_ofmtl;
            % update lambda
            ofmtl_lambda = mocha_lambda;
        else
            tic
            [W, Sigma, rmse, ~] = ofmtl(Xtrain, Ytrain, Xtest, Ytest, Sigma, W, ofmtl_lambda, opts);
            err_ofmtl{trial}(i+1) = rmse(end);
            temp = toc
            temp_ofmtl = temp_ofmtl + temp;
            time_ofmtl{trial}(i+1) = temp_ofmtl;
        end
              
    end
    clear Xtrain;
    clear Ytrain;    
    
end