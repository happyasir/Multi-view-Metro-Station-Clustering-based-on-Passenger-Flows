function  [TTloss,UU,B,w,w1] = sparse_weighted_clustero(Z,eta,lambda,lambda1,lambda2,L0,U0,B0,w0,niter,niter2,min_w)
[nnode,~,nlayer] = size(Z);
ncluster = size(B0,1);
BB = [];
UU = [];
TTloss = [];
normZ = [];

for m = 1:nlayer
    normZ(m) = norm(Z(:,:,m),'fro').^2;
end
flag = 1;
i = 1;
while i < niter
    I = eye(nlayer);
    B1 = [];
    Z1 = [];
    flag1 = 0;
    if i ==1
        U = U0;
        B = B0;
        w = w0;
        w1 = w./normZ';
    end
    
    for m = 1:nlayer
     B(:,:,m) = B(:,:,m).*sqrt((U'*Z(:,:,m)*U)./(U'*U*B(:,:,m)*U'*U));
  %   B(:,:,m) = update_B_proximal(Z(:,:,m),U,B(:,:,m),w1(m),niter2,eta,lambda1);
    end
   [U,flag0,total_loss] = update_U_proximal1(Z,U,B,w1,niter2,eta,lambda,0);
%    figure;
%    plot(total_loss)
    w = update_wo(Z,U,B,L0,min_w,lambda2);
    w1 = w./normZ';
    UU(:,:,i) = U;
   % calculate total loss
    for k = 1:nlayer
        B1 = blkdiag(B1,squeeze(B(:,:,k)));
        Z1 = blkdiag(Z1,squeeze(sqrt(w1(k)).*Z(:,:,k)));
    end
    TTloss(i) = sum(sum((Z1-kron(diag(sqrt(w1)),U)*B1*kron(eye(nlayer),U)').^2))/2 ...
        +norm(U,1)*lambda + lambda2*w'*L0*w;
    if i >1
       norm_diff_U(i) = norm(U-UU(:,:,i-1),'fro');
    if (norm_diff_U(i)+norm_diff_U(i-1)) <0.5e-3
    flag = 0;
    else 
        i = i+1;
    end
    else
        i = i+1;
    end  
end
lose1(1) = sum(sum((Z1-kron(diag(sqrt(w1)),U)*B1*kron(eye(nlayer),U)').^2))/2;
lose1(2) = norm(U,1);
lose1(3) = w'*L0*w;

end
