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
mu_carci = 0.87; % carcinoma 0.392 @30keV; 0.844 @20keV;
mu_ca    = 1.2;   % calcification

fin         = fopen('/media/dril/BackupPlus/lesions/mass_327139370_182.raw');
lesion_size = 182;
I   = fread(fin, 182*182*182,'uint8=>uint8');
Z   = reshape(I, 182, 182, 182);
Z   = downsample3(Z, 5); 
Z   = imbinarize(Z);



for phantomindex=1:1
    
     load(strcat(['/media/dril/ubuntudata/DBT-NEW/attenuation_values_cropped/', int2str(phantomindex), '.mat']));
     x = head;
% 
%     %downsample to run faster
%     down   = 2;
%     x      = downsample3(x, down);
    
    lesion = zeros(size(x));
    
    %load(strcat(['/media/dril/ubuntudata/DBT-NEW/gan-90-projections/sart_', int2str(phantomindex), '.mat']));
    %x = xartt;
    
    % Insert lesion at random location
    lesion_flag = 10;%rand;
    if lesion_flag > 0.3
        disp('Lesion inserted');
        disp(phantomindex);
        
        lesion_x = randi([200, 300], 1, 1);
        lesion_y = randi([60,  160], 1, 1);
        lesion_z = randi([40,  120], 1, 1);
        
        lesion(lesion_x-18:lesion_x+17, lesion_y-18:lesion_y+17, lesion_z-18:lesion_z+17) = Z;
        x(lesion==1) = mu_carci;
    end
    
    
    
    
    %===============================    
    %Define the object geometry
    %===============================
    [nx,ny,nz] = size(x);                         % phantom dimensions
    dx   = 0.02*down; dy=0.02*down; dz=0.02*down; % in cm, phantom pixel sizes
    xfov = dx*nx; % 20;
    yfov = dy*ny;
    zfov = dz*nz;

    offset_y = -ny/2;    % offset of the object ctr in pixels for the y-dimension; 0 for full cone, -ny/2 for half cone
    d_objbottom_det = 0; % in cm,the distance from the bottom of the object to the center detector.
                         % Value "0" means the object is places right on the
                         % detector.

    % ============================
    % Define the scanner geometry
    % ============================
    % for arc trajectory
     dso = 60.5; %in cm: dist. from the source to the rotation center
     dod = 4.5;  %in cm: dist. from the rotation center to the detector
     dsd = dso+dod; %in cm: dist. from source to the detector

     %91.666666
     orbit = 50;  % angular span
     na    = 25;         % number of projection views
     ds    = dx;         % in cm; detector pixel size
     dt    = dx;
     
     %calculate the length and width of the detector so it is large enough to cover the
     %projection views from the most oblique angles.
     %costheta=cos(orbit/2*pi/180); sintheta=sin(orbit/2*pi/180);
     %sfov = ((dso*costheta+dod)*(xfov/2+dso*sintheta)/(dso*costheta+offset_z*dz-zfov/2) - dso*sintheta)*2;
     %tfov = yfov*(dso*costheta+dod)/(dso*costheta+dod-zfov);
     ns = 800;%ceil(sfov/ds);
     nt = 300;%ceil(tfov/dt);

     offset_s = 0; %detector center offset along the 's' direction in pixels relative to the tube rotation center
     offset_t = -nt/2; %detector center offset along the 't' direction in pixels relative to the tube rotation center
     offset_z = (dod - (zfov/2 + d_objbottom_det - dz/2 ))/dz; %in pixels, offset of the object ctr to the rotation ctr in the z direction;

     %==============================
     %Create DBT projection views
     %==============================
     btg = bt_geom('arc', 'ns', ns, 'nt', nt, 'na', na, ...
            'ds', ds, ...%'dt', dv, ... defautly dt = -ds;
            'down', 1, ...
            'orbit', orbit,...
            'offset_s', 0, ... % quarter detector 
            'offset_t', offset_t, ...
            'dso', dso, 'dod', dod, 'dfs',inf);  

    ig = image_geom('nx', nx, 'ny',ny, 'nz', nz, 'dx',dx, 'dz', dz,...
           'offset_y', offset_y,'offset_z', offset_z,  'down', 1); 

    Gt = Gtomo_syn(btg,ig); %generate system Matrix
    
    
    calcification_flag = rand;
    % add a spherical lesion to the phantom
    if(0)
        for k=1:5
            rx=dx*2; ry=dy*2; rz=dz*2;
            % lesion radius
            % define the geometric properties of a sphere
            ell          = [ig.x(182) ig.y(128) ig.z(150) rx ry rz 0 0 1];      
            lesion       = ellipsoid_im(ig,ell); % generate the lesion volumes
            x(lesion==1) = mu_carci;   

            % lesion locations
            lesion_x = randi([200, 300], 1, 1);
            lesion_y = randi([80,  140], 1, 1);
            lesion_z = randi([40,  120], 1, 1);

            ell          = [ig.x(lesion_x) ig.y(lesion_y) ig.z(lesion_z) rx ry rz 0 0 1];      
            lesion       = ellipsoid_im(ig, ell); % generate the lesion volumes
            x(lesion==1) = mu_ca;     
        end
    end

    nview = length(Gt);
    g     = zeros(btg.ns,btg.nt,nview);
    for i=1:nview
       %tic   
       g(:,:,i) = Gt{i}*permute(x,[1 3 2]);   
       %toc
    end
    
    %add poisson noise to the projections
    if(0)
        [proj_noi, g_noi, g_noi_nonoise, I0] = insert_noise(0.75, g, na);
        %save(strcat(['/media/dril/ubuntudata/DBT-NEW/gan-90-projections/proj_noi_',     int2str(phantomindex), '.mat']), 'proj_noi', 'I0');
        %save(strcat(['/media/dril/ubuntudata/DBT-NEW/gan-90-projections/g_noi_',        int2str(phantomindex), '.mat']), 'g_noi');
        save(strcat(['/media/dril/ubuntudata/DBT-NEW/gan-90-projections/g_noi_sart_',    int2str(phantomindex), '.mat']), 'g_noi_nonoise');
        
%         [proj_noi, g_noi, I0] = insert_noise(1.5, g, na);
%         save(strcat(['/media/dril/ubuntudata/DBT-NEW/projections-noise/proj_noi_', int2str(phantomindex), '_2.mat']), 'proj_noi', 'I0');
%         save(strcat(['/media/dril/ubuntudata/DBT-NEW/projections-noise/g_noi_',    int2str(phantomindex), '_2.mat']), 'g_noi');
%         
%         [proj_noi, g_noi, I0] = insert_noise(3, g, na);
%         save(strcat(['/media/dril/ubuntudata/DBT-NEW/projections-noise/proj_noi_', int2str(phantomindex), '_3.mat']), 'proj_noi', 'I0');
%         save(strcat(['/media/dril/ubuntudata/DBT-NEW/projections-noise/g_noi_',    int2str(phantomindex), '_3.mat']), 'g_noi');
    end
end

