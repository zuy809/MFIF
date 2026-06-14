function warp_out=param_update(warp_in,delta_p,transform)

if strcmp(transform,'homography')
    delta_p=[delta_p; 0];
    warp_out=warp_in + reshape(delta_p, 3, 3);
    warp_out(3,3)=1;
end

if strcmp(transform,'affine')

    warp_out(1:2,:)=warp_in(1:2,:)+reshape(delta_p, 2, 3);
    warp_out=[warp_out;zeros(1,3)];
    warp_out(3,3)=1;
end

if strcmp(transform,'translation')
    warp_out =warp_in + delta_p;
end

if strcmp(transform, 'euclidean')

    theta = sign(warp(2,1))*acos(warp(1,1))+delta_p(1);
    tx = warp_in(1,3)+delta_p(2);
    ty = warp_in(2,3)+delta_p(3);
    warp_out = [cos(theta) -sin(theta) tx;...
                sin(theta) cos(theta) ty;...
                    0         0        1];
                
end
