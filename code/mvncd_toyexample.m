clear; clc;
load('data_mvncd.mat');
select_layer = [2];
Z_cluster = Z_TWSNMF(:,:,select_layer);
nlayer = size(Z_cluster,3);

L0 = corr(SG_base(:,select_layer));

niter = 40;
ncluster =4;
U0 = rand(nboard,ncluster);
B0 = rand(ncluster,ncluster,nlayer);
w0 = ones(nlayer,1)./nlayer;

niter2 = 50;
eta = 1.1;
lambda = 0.0001;
lambda1 = 0.001;
lambda2 = 1;
min_w= 1/nlayer;
%%
[TTloss,UU,B,w,w1] = sparse_weighted_clustero(Z_cluster,eta,lambda,lambda1,lambda2,L0,U0,B0,w0,niter,niter2,min_w)
U = UU(:,:,end);
plot(TTloss);