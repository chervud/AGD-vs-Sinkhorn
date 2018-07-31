function [iter,time]=Sinkhorn(r,c,epsilon)

%Sinkhorn's algorithm from [Altshuler et al, 2017], Alg.3

% r -- vector of first measure
% c -- vector of second measure

tic;

maxIter = 20000;

n = size(r,1);

global C; %cost matrix
gamma = epsilon/4/log(n); %regularization parameter

A = exp(-C/gamma);
A_0 = A;
A = A/sum(sum(A));

max_el = max(max(C));

k=0;
x = zeros(n,1);
y = zeros(n,1);

while (norm(r - sum(A,2),1) + norm(c - sum(A,1)',1) > epsilon/8/max_el && k < maxIter)
    str = ['k = ',num2str(k),', residual = ',num2str(norm(r - sum(A,2),1) + norm(c - sum(A,1)',1))];
    disp(str);
    k = k+1;
    if mod(k,2) == 1
        x = x + log(r./sum(A,2));
    else
        y = y + log(c./(sum(A,1)'));
    end
    C_new = -C/gamma+x*ones(1,n)+ones(n,1)*y';
    A = exp(C_new);
    if k >= 500
        break;
    end
end

str = ['average time per iteration ',num2str(toc/k)];
disp(str); %print current iteration number   

iter = k;
time = toc;

end