% example with set of written digits. Use digits.mat as input data 

data = importdata('mnist.mat');
digits = testX';

%size of an image
scale = 1; %multiplier for the number of pixels
m = 28*scale; %m x m - 2-dim grid, size of the support = m^2
%size of the support of the measures
n = m*m;

% calculate transport cost matrix
global C;
disp('computeDistanceMatrixGrid');
C = computeDistanceMatrixGrid(m);
disp('C.*C');
C = C.*C;
disp('C / median(C(:))');
C = C / median(C(:)); %normalize cost matrix to set natural scale for \gamma

%initialize parameters
epsilon = 0.05;

runs = 1; %number of runs of the experiment
results_rand_images_m_50_eps_01 = zeros(6,runs);


for k = 1:1:runs
%load images from MNIST
%generate image number
i = ceil(rand*size(digits,2));
j = ceil(rand*size(digits,2));

aa = im2double(digits(:,i));
bb = im2double(digits(:,j));
aa = aa/sum(aa);
bb = bb/sum(bb);
aa = reshape(aa, m/scale, m/scale);
aa = my_im_resize(scale,scale,aa);
a = reshape(aa, m*m, 1);
bb = reshape(bb, m/scale, m/scale);
bb = my_im_resize(scale,scale,bb);
b = reshape(bb, m*m, 1);
b = b/sum(b);

I = (a==0);
a(I) =  0.000001;
I = (b==0);
b(I) =  0.000001;


a = a/sum(a);
b = b/sum(b);



%random images
% disp('generate images');
% num_of_non_zeros = round(n*0.4);
% a = [rand(num_of_non_zeros,1);zeros(n-num_of_non_zeros,1)];
% b = [zeros(n-num_of_non_zeros,1);rand(num_of_non_zeros,1)];
% I = (a==0);
% a(I) =  0.0001;
% I = (b==0);
% b(I) =  0.0001;
% a = a/sum(a);
% b = b/sum(b);
% a = a(randperm(length(a)));
% b = b(randperm(length(b)));


%%
% %Run Sinkhorn
disp('start Sinkhorn');
[iter,time] = Sinkhorn(a,b,epsilon); 
%results_rand_images_m_50_eps_01(1,k) = iter;
%results_rand_images_m_50_eps_01(2,k) = time;


%%
%Run APDAGD for log_sum_exp
[iter,time] = PDASTM(a,b,epsilon); 
results_rand_images_m_50_eps_01(3,k) = iter;
results_rand_images_m_50_eps_01(4,k) = time;


end
