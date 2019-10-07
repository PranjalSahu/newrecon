%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Example code to simulate DBT projection views.
% 
% Author: Rongping Zeng, FDA/CDRH/OSEL/DIDSR, 
% Contact: rongping.zeng@fda.hhs.gov
% Feb. 2018
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%attenuation properties of breast tissue: Jonn and
%Yaffe-1987-pmb-v32-p678-Table1
mu_fg    = 0.802; % 0.378 @30keV;  0.802 @20keV;
mu_adp   = 0.456; % 0.264 @30keV; 0.456 @20keV
%mu_carci = 0.844; % carcinoma 0.392 @30keV; 0.844 @20keV;
mu_carci = 0.89;
mu_ca    = 1.2;   % calcification


load /media/dril/ubuntudata/DBT-NEW/attenuation_values_cropped/153.mat;
%load /media/dril/ubuntudata/DBT-NEW/deeplearning_output/153.mat;
%load /media/dril/ubuntudata/DBT-NEW/deeplearning_output/153_3_hann50.mat

x = head;

%downsample to run faster
%down = 2;
%x    = downsample3(x, down);


%x = xartt1(:, :, :, 5);

%load head.mat;

%===============================    
%Define the object geometry
%===============================

% 0.0255

[nx,ny,nz] = size(x);                         % phantom dimensions
dx   = 0.0255; dy = dx; dz = dx;      % in cm, phantom pixel sizes
xfov = dx*nx; %20;
yfov = dy*ny;
zfov = dz*nz;

offset_y        = -ny/2; % offset of the object ctr in pixels for the y-dimension; 0 for full cone, -ny/2 for half cone
d_objbottom_det = 0;     % in cm,the distance from the bottom of the object to the center detector.
                         % Value "0" means the object is places right on the
                         % detector.

%============================
%Define the scanner geometry
%============================
% for arc trajectory
% 90 angle -> 91.666666  (45 projections)
% 50 angle -> 50         (25 projections)
% 70 angle -> 70.833333  (35 projections)

 dso = 60.5;      % in cm: dist. from the source to the rotation center
 dod = 4.5;       % in cm: dist. from the rotation center to the detector
 dsd = dso + dod; % in cm: dist. from source to the detector
 
 %91.666666
 orbit = 50;  % angular span
 na    = 25;  % number of projection views
 
 ds = dx;     % in cm; detector pixel size
 dt = dx;
 %calculate the length and width of the detector so it is large enough to cover the
 %projection views from the most oblique angles.
 %costheta=cos(orbit/2*pi/180); sintheta=sin(orbit/2*pi/180);
 %sfov = ((dso*costheta+dod)*(xfov/2+dso*sintheta)/(dso*costheta+offset_z*dz-zfov/2) - dso*sintheta)*2;
 %tfov = yfov*(dso*costheta+dod)/(dso*costheta+dod-zfov);
 ns = 1600; % ceil(sfov/ds);
 nt = 600;  % ceil(tfov/dt);

 offset_s = 0;     % detector center offset along the 's' direction in pixels relative to the tube rotation center
 offset_t = -nt/2; % detector center offset along the 't' direction in pixels relative to the tube rotation center
 offset_z = (dod - (zfov/2 + d_objbottom_det - dz/2 ))/dz; % in pixels, offset of the object ctr to the rotation ctr in the z direction;

 %==============================
 %Create DBT projection views
 %==============================
 btg = bt_geom('arc', 'ns', ns, 'nt', nt, 'na', na, ...
		'ds', ds, ...%'dt', dv, ... defautly dt = -ds;
		'down', 1, ...
        'orbit', orbit,...
        'orbit_start', 0,...
		'offset_s', 0, ... % quarter detector 
		'offset_t', offset_t, ...
   		'dso', dso, 'dod', dod, 'dfs', inf);  

ig = image_geom('nx', nx, 'ny',ny, 'nz', nz, 'dx',dx, 'dz', dz,...
       'offset_y', offset_y,'offset_z', offset_z,  'down', 1); 

Gt = Gtomo_syn(btg, ig); %generate system Fatrix

% Add a realistic lesion to the phantom
% fin = fopen('/media/dril/BackupPlus/lesions/mass_327139370_182.raw');
% lesion_size = 182;
% I   = fread(fin, 182*182*182,'uint8=>uint8');
% Z   = reshape(I, 182, 182, 182);
% Z   = downsample3(Z, 5); 
% Z   = imbinarize(Z);
% lesion = zeros(size(x));

% lesion candidates
% [182, 128, 140], [212, 128, 140], [212, 128, 149], [212, 128, 143]

% lesion_x = 182;
% lesion_y = 128;
% lesion_z = 120;
% lesion(lesion_x-18:lesion_x+17, lesion_y-18:lesion_y+17, lesion_z-18:lesion_z+17) = Z;
%x(lesion==1) = mu_carci;



% add a spherical lesion to the phantom
% if(1)
%     rx=dx*12; ry=dy*12; rz=dz*12;
%     % lesion radius
%     % define the geometric properties of a sphere
%     % ell=[xctr, yctr, zctr, rx, ry, rz, alpha, beta, attenuation];
%     
%     % lesion locations
%     %
%     %
%     %
%     ell          = [ig.x(182) ig.y(128) ig.z(150) rx ry rz 0 0 1];      
%     lesion       = ellipsoid_im(ig,ell); % generate the lesion volumes
%     x(lesion==1) = mu_carci;             % assign the attnuation value to lesion voxels.
%                                          % This lesion is bright so will be highly
%                                          % visuable in the recosntructed DBT volume.
% end

nview = length(Gt);
g     = zeros(btg.ns, btg.nt, nview);

for i=1:nview
   %tic   
   g(:,:,i) = Gt{i}*permute(x,[1 3 2]);   
   %toc
end

%===========================
%add Poisson noise
%===========================
g_noi    = g;

%I0       = 10^5/na;
I0       = na*(3*10^4/25); % distriburte the entire dose evenly to each proejction view.

disp('I0 value is');
disp(I0);

proj     = I0*exp(-g);
proj_noi = proj;

g_noi_nonoise                       = log(I0)-log(proj_noi); % convert back to line integrals 
g_noi_nonoise(g_noi_nonoise < 0)    = 0;

%add poisson noise to the projections
if(0)
    proj_noi                    = poissrnd(proj);        % poisson(proj);%
    proj_noi(find(proj_noi==0)) = 1; 
    
    g_noi                       = log(I0)-log(proj_noi); % convert back to line integrals 
    g_noi(g_noi<0)              = 0;    
end

if(0)
    save proj_noi.mat proj_noi I0;
    save g_noi1.mat g_noi;
    save g_noi_nonoise.mat g_noi_nonoise;
    save head.mat x;
end


