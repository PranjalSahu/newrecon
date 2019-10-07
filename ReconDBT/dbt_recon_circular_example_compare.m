%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Example code of using the DBT reconstruction functions for arc DBT geometry.
% The projection data in this code was created by "gen_dbtproj_example.m".
%
% Author: Rongping Zeng, FDA/CDRH/OSEL/DIDSR, 
% Contact: rongping.zeng@fda.hhs.gov
% Feb. 2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% load in the projection views.
% "g_noi.mat"    is for FBP and SART reconstruction.
% "proj_noi.mat" is for ML reconstruction.

load proj_noi.mat; % the variable is 'proj_noi'.
load g_noi.mat;    % the variable is 'g_noi'.


% Good Cases to see
% 162, 163

load '/media/dril/ubuntudata/DBT-NEW/gan-90-projections/projections/g_noi_noise_166.mat';
a        = g_noi_nonoise;
[s1, x1] = do_recon(a, 91.6666);


load '/media/dril/ubuntudata/DBT-NEW/gan-90-projections/predictions/gan-model1/26_prediction.mat';
b        = prediction;
b        = double(b);
[s2, x2] = do_recon(b, 91.6666);


load '/media/dril/ubuntudata/DBT-NEW/gan-90-projections/projections/g_noi_166.mat';
c        = g_noi;
c        = c(:, :, 11:35); 
[s3, x3] = do_recon(c, 50);

load head.mat;


t1 = (s1(:, :, sliceindex) + s1(:, :, sliceindex+1) + s1(:, :, sliceindex-1))/3;
t2 = (s2(:, :, sliceindex) + s2(:, :, sliceindex+1) + s2(:, :, sliceindex-1))/3;
t3 = (s3(:, :, sliceindex) + s3(:, :, sliceindex+1) + s3(:, :, sliceindex-1))/3;
imshow([t1 t2 t3]);

function [xfbp, xartt] = do_recon(g, orbit_angle)
     nview = size(g, 3);

     for i=1:nview
       temp = g(:, :, i);
       t    = graythresh(temp);
       temp(temp < t) = 0;
       g(:, :, i) = temp;  
    end

     dso = 60.5; %in cm: dist. from the source to the rotation center
     dod = 4.5;  %in cm: dist. from the rotation center to the detector
     dsd = dso+dod; %in cm: dist. from source to the detector


     orbit = orbit_angle;     % in degree: angular span
     na = size(g,3); % number of projection views
     ds = 0.04;      % in cm: detector element pixel size in the 's' direction; 
     dt = 0.04;      % in cm: detector element pixel size in the 's' direction; 
                     % 's', the x-ray tube moving direction, positive pointing toward right.           
                     % 't', the perpendicular direction to 's' direction, positive pointing toward the nipple.            

     ns       = size(g,1); % number of detector elements in the 's' direction
     nt       = size(g,2); % number of detector elements in the 's' direction
     offset_s = 0;         % detector center offset along the 's' direction in pixels relative to the tube rotation center
     offset_t = -nt/2;     % detector center offset along the 't' direction in pixels relative to the tube rotation center
     d_objbottom_det = 0;  % in cm,the distance from the bottom of the object to the center detector.
    

    nrx      = 400;
    nry      = 224;
    drx      = 0.04;  
    dry      = drx; 
    drz      = 0.04; 
    nrz      = 160; 
    offset_x = 0; %in pixels
    offset_y = -nry/2;% in pixels. 0 for full cone, -nry/2 for half cone
    zfov     = nrz*drz;
    offset_z = (dod - (zfov/2 + d_objbottom_det-drz/2))/drz; %in pixels: offset of the volume ctr to the rotation ctr in the z direction; 
                                
 
     %===================
     %Reconstruction
     %===================

     %Generate the system matrix

     igr = image_geom('nx', nrx, 'ny',nry, 'nz', nrz, 'dx',drx, 'dz', drz,...
           'offset_y', offset_y,'offset_z', offset_z,'down', 1); 


     btg = bt_geom('arc', 'ns', ns, 'nt', nt, 'na', na, ...
            'ds', ds, ...%'dt', dv, ... defautly dt = -ds;
            'down', 1, ...
            'orbit', orbit,...
            'offset_s', 0, ...  
            'offset_t', offset_t, ...
            'dso', dso, 'dod', dod, 'dfs',inf);  
    
     Gtr = Gtomo_syn(btg, igr);
     
     % FBP reconstruction
     xfbp = fbp_dbt(Gtr, btg, igr, g, 'hann75');
    
     xartt = xfbp;
     % SART reconstruction
     %xbp = BP(Gtr, g); % initialization for SART
     %disp(size(xbp));
     
     %[xartt, costart, diff_image_final, back_proj_images] = SART_dbt(Gtr,  g, xbp, 5, 0.9, 1);
end

