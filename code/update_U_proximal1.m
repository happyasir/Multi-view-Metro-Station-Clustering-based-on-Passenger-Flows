function [U,flag1,total_loss] = update_U_proximal1(Z,U0,B,w,niter,eta,lambda,normal_flag)
[nnode,~,nlayer] = size(Z);
[ncluster,~,~] = size(B);

I = eye(nlayer);
B1 = [];
Z1 = [];
flag1 = 0;
for i = 1:nlayer
    B1 = blkdiag(B1,squeeze(B(:,:,i)));
    Z1 = blkdiag(Z1,squeeze(sqrt(w(i)).*Z(:,:,i)));
end
UU = zeros(nnode,ncluster,niter);
UU(:,:,1) = U0;
U = U0;

total_loss(1) = sum(sum((Z1-kron(diag(sqrt(w)),U)*B1*kron(eye(nlayer),U)').^2))/2 ...
    +norm(U,1)*lambda;

nn = 0;
L(1) = 2;
for iter = 2:niter
    U = UU(:,:,iter-1);
    L(iter) = L(iter-1)*eta^(nn-1);
    %L(iter) = L(iter-1)*eta;
    G_k = total_loss(iter-1);
    for m = 1:nlayer
        Delta_Um(:,:,m) = w(m)*(U*B(:,:,m)*U'*U*B(:,:,m)'+U*B(:,:,m)'*U'*U*B(:,:,m)...
            -Z(:,:,m)*U*B(:,:,m)'-Z(:,:,m)'*U*B(:,:,m));
    end
    Delta_U = squeeze(sum(Delta_Um,3));
    U_bar = U-Delta_U./L(iter);
    U1 = U_bar-lambda./L(iter).*sign(U_bar);
    U1(find(abs(U_bar)<lambda/L(iter))) = 0;
    U1(find(U1<0)) = 0;
    if normal_flag ==1
        U1(find(U1>1)) = 1;
    end
    U = U1;
    total_loss(iter) = sum(sum((Z1-kron(diag(sqrt(w)),U)*B1*kron(eye(nlayer),U)').^2))/2 ...
        +norm(U,1)*lambda;
    %  G(iter)= total_loss(iter-1)-lambda*norm(UU(:,:,iter-1),1) + sum(sum(Delta_U.*(U-UU(:,:,iter-1))))...
    %        + norm(U-UU(:,:,iter-1),'fro').^2/2*L(iter)+lambda*norm(U,1);
    G(iter)= total_loss(iter-1) + sum(sum(Delta_U.*(U-UU(:,:,iter-1))))...
        + norm(U-UU(:,:,iter-1),'fro').^2/2*L(iter);
    if G(iter)>=total_loss(iter)
        flag22 = 0;
        UU(:,:,iter) = U;
    else
        nn = nn+1;
    end
end
end