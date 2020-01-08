function [proj_noi, g_noi, g_noi_nonoise, I0] = insert_noise(dose_level, g, na)

        I0        = na*(3*10^4/25);
        %I0       = dose_level*10^5/na;                       % distribute the entire dose evenly to each proejction view
        proj      = I0*exp(-g);
        proj_noi1 = proj;
        proj_noi  = proj_noi1;
        
        proj_noi                     = poissrnd(proj);        % poisson(proj);%
        
        proj_noi(find(proj_noi==0))  = 1; 
        g_noi                        = log(I0)-log(proj_noi); % convert back to line integrals 
        g_noi(g_noi < 0)             = 0;
        
        
        proj_noi1(find(proj_noi1==0))  = 1; 
        g_noi_nonoise                  = log(I0)-log(proj_noi1);
        g_noi_nonoise(g_noi_nonoise<0) = 0;
end