function G = image_jacobian(gx, gy, jac, nop)

[h,w]=size(gx);

if nargin<4
    error('Not enough input arguments');
end

gx=repmat(gx,1,nop);
gy=repmat(gy,1,nop);

G=gx.*jac(1:h,:)+gy.*jac(h+1:end,:);
G=reshape(G,h*w,nop);