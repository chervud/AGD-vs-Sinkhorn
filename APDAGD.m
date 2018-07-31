function [iter,time]=PDASTM(r,c,epsilon)




%Input parameters
% r -- vector of first measure
% c -- vector of second measure

tic;

maxIter = 20000;

%% define dimensions
n = size(r,1); %dimension of the vector which represents the measure

global C; %cost matrix
max_el = max(max(C));
gamma = epsilon/3/log(n);


X = zeros(n,n); %output primal variable

    
    %initiate variables for APDAGD
	%x is \lambda
	%y is \eta
	%z is \zeta
    psi_x = zeros(1, maxIter); %objective values at points x_k which are accumulated in the model of the function. 
    grad_psi_x = zeros(2*n, maxIter); %gradient values at x_k which are accumulated in the model of the function
    psi_y = zeros(1, maxIter); %objective values at points y_k for which usually the convergence rate is proved
    gap = zeros(1, maxIter);
    A = zeros(maxIter, 1); % init array of A_k
    L = zeros(maxIter, 1); %init array of L_k
    x = zeros(2*n, maxIter); %init array of points x_k where the gradient is calculated (lambda)
    y = zeros(2*n, maxIter); %init array of points y_k for which usually the convergence rate is proved (eta)
    z = zeros(2*n, maxIter); %init array of points z_k. this is the Mirror Descent sequence. (zeta)    
    
    
    %set initial values for APDAGD
    L(1, 1) = 1; %set L_0
    
    %set starting point for APDAGD
    x(:, 1) = zeros(2*n, 1); %basic choice, set x_0 = 0 since we use Euclidean structure and the set is unbounded
    
    disp('starting APDAGD');
    
    
    %main cycle of APDAGD
    for k = 1 : 1 : (maxIter - 1)
        
        str = ['k=',num2str(k)];
        disp(str); %print current iteration number        
        
        %init for inner cycle
        flag = 1; %flag of the end of the inner cycle
        j = 0; %corresponds to j_k
        while flag > 0
            disp(j); %print current inner iteration number
                       
            L_t = 2^(j-1)*L(k, 1); %current trial for L
                
            a_t = (1  + sqrt(1 + 4 * L_t * A(k, 1)) )/ 2 / L_t ; % trial for calculate a_k as solution of quadratic equation explicitly
            A_t = A(k, 1) + a_t; %trial of A_k
            tau = a_t / A_t; %trial of \tau_{k}
            
            x_t = tau * z(:, k) + (1 - tau) * y(:, k); %trial for x_k
            
            %calculate trial oracle at xi           
            %calculate function \psi(\lambda,\mu) value and gradient at the trial point of x_{k}
            lambda = x_t(1:n,1);
            mu = x_t(n+1:2*n,1);            
            C_new = -C-lambda*ones(1,n)-ones(n,1)*mu';
            X_lambda = exp(C_new/gamma);
            sum_X = sum(sum(X_lambda));
            X_lambda = X_lambda/sum_X;
            grad_psi_x_t(:,1) = zeros(2*n,1);
            grad_psi_x_t(1:n,1) = r - sum(X_lambda,2);
            grad_psi_x_t(n+1:2*n,1) = c - sum(X_lambda,1)';
            psi_x_t = lambda'*r + mu'*c + gamma*log(sum_X);            
            
            %update model trial
            z_t = z(:, k) - a_t * grad_psi_x_t; %trial of z_k 
            
            y_t = tau * z_t + (1 - tau) * y(:, k); %trial of y_k
            
            %calculate function \psi(\lambda,\mu) value and gradient at the trial point of y_{k}
            lambda = y_t(1:n,1);
            mu = y_t(n+1:2*n,1);            
            C_new = -C-lambda*ones(1,n)-ones(n,1)*mu';
            Z = exp(C_new/gamma);
            sum_Z = sum(sum(Z));
            psi_y_t = lambda'*r + mu'*c + gamma*log(sum_Z);
            
            l = psi_x_t + mtimes(grad_psi_x_t',y_t - x_t) + L_t / 2 * norm(y_t - x_t,2)^2;%calculate r.h.s. of the stopping criterion in the inner cycle
            
            if psi_y_t <= l %if the stopping criterion is fulfilled
                flag = 0; %end of the inner cycle flag
                x(:, k + 1) = x_t; %set x_{k+1}
                y(:, k + 1) = y_t; %set y_{k+1}
                z(:, k + 1) = z_t; %set y_{k+1}
                psi_y(1, k + 1) = psi_y_t; %save psi(y_k)                
                A(k + 1, 1) = A_t; %set A_{k+1}
                L(k + 1, 1) = L_t; %set L_{k+1}
                X = tau * X_lambda + (1-tau) * X; %set primal variable             
            end
            j = j + 1;
        end
    
        %check stopping criterion
        X_hat = round_matrix(X,r,c); %Apply rounding procedure from [Altshuler et al, 2017]
        error_constr = sum(sum(C.*(X_hat-X))); %error in the equality constraints
        disp(['current error = ',num2str(error_constr),', goal = ',num2str(epsilon/6/max_el)]);
        if error_constr < epsilon/6/max_el
            break
        end
        
        
    end
    
    str = ['average time per iteration ',num2str(toc/k)];
    disp(str); %print current iteration number           

    iter = k;
    time = toc;

end
