function warp=next_level(warp_in, transform, high_flag)


warp=warp_in;
if high_flag==1
    if strcmp(transform,'homography')
        warp(7:8)=warp(7:8)*2;
        warp(3)=warp(3)/2;
        warp(6)=warp(6)/2;
    end
    
    if strcmp(transform,'affine')
        warp(7:8)=warp(7:8)*2;
        
    end
    
    if strcmp(transform,'translation')
        warp = warp*2;
    end
    
    if strcmp(transform,'euclidean')
        warp(1:2,3) = warp(1:2,3)*2;
    end
    
end

if high_flag==0
    if strcmp(transform,'homography')
        warp(7:8)=warp(7:8)/2;
        warp(3)=warp(3)*2;
        warp(6)=warp(6)*2;
    end
    
    if strcmp(transform,'affine')
        warp(7:8)=warp(7:8)/2;
    end
    
    if strcmp(transform,'euclidean')
        warp(1:2,3) = warp(1:2,3)/2;
    end
    
    if strcmp(transform,'translation')
        warp = warp/2;
    end
    
end