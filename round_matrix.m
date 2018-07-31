function [X_hat]=round_matrix(X_hat_k,r,c)

one = ones(size(X_hat_k,1),1);

x = r./(X_hat_k*one);
x = arrayfun(@(t) min(t,1),x);
F_1 = bsxfun(@times,X_hat_k',x')';
%F_1 = diag(x)*X_hat_k;

y = c./(F_1'*one);
y = arrayfun(@(t) min(t,1),y);
F_2 = bsxfun(@times,F_1,y');
%F_2 = F_1*diag(y);

err_r = r - F_2*one;
err_c = c - F_2'*one;

X_hat = F_2 + err_r*err_c'/norm(err_r,1);



end