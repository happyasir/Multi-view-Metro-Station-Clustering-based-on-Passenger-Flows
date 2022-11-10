function w1 = update_wo(Z,U,B,L0,min_w,lambda2)
nlayer = size(Z,3);
for m = 1:nlayer
    S(m) = norm(Z(:,:,m)-U*B(:,:,m)*U','fro')^2/2./norm(Z(:,:,m),'fro').^2;
end
w1 = quadprog(L0.*lambda2,S,[],[],ones(1,nlayer),1,ones(nlayer,1).*min_w,ones(nlayer,1));
end