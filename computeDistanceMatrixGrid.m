
% computes transport cost matrix of squared Euclidean distances

function [ M ] = computeDistanceMatrixGrid(N)
%N - size of support
A = zeros(N ^ 2, 2); % generate two column vectors of length N^2
iter = 0;
for i = 1 : 1 : N
    for j = 1 : 1 : N
        iter = iter + 1;
        A(iter, 1) = i;
        A(iter, 2) = j;        
    end
end
% Now A contains all possible matchings of N elements
M = pdist2(A, A);
end

