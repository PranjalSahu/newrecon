% Downsample around 4.7 times to get equivalent resolution from the real
% DBT projection
% fid = fopen('/media/dril/ubuntudata/DBT_recon_data/CE12/Projections/Projections_Renamed_Seg/CE12.3584x1800.0018.raw');
% c   = fread(fid, 3584*1800, 'float');
% cb  = reshape(c, [3584, 1800]);


% For storing the raw files
% fid = fopen('s5.raw','w+');
% cnt = fwrite(fid, cb1,'float');
% fclose(fid);


% For reading the real DBT projections
filepath = '/media/dril/ubuntudata/DBT_recon_data/CE12/';
sx_a   = 1800;
sy_a   = 3584;
slices = 46;
sx_b   = 2600;
sy_b   = 1300;
volume_name     = 'CE-12_2600x1300_46.raw';
offdetector_height = 35;
anglefile       = strcat(filepath, 'angles.ini');
projections_dir = strcat(filepath, 'Projections/Projections_Renamed_Seg');
volume_path     = strcat(filepath,  volume_name);

%fid    = fopen(anglefile, 'r');
%angles = fread(fid, 25, 'float');
%angles = angles';


noise_projections = zeros(sy_a/2, sx_a/2, 25, 'double');
files             = dir(projections_dir);

for t=3:27
  disp(strcat(files(t).folder, '/', files(t).name))
  fid = fopen(strcat(files(t).folder, '/', files(t).name), 'r');
  c   = fread(fid, sx_a*sy_a, 'float');
  cb  = reshape(c, [sy_a, sx_a]);
  %cb = cb';
  temp    = graythresh(cb);
  cb(cb < temp) = 0;
  cb = imresize(cb, 0.5);
  noise_projections(:, :, 26-(t-2)) = flip(cb, 2);
end

g = noise_projections;
%noise_projections = single(noise_projections);
nview = size(g, 3);

%==================================
%User Defines the scanner geometry
%==================================
 
 dso = 60.5; %in cm: dist. from the source to the rotation center
 dod = 4.5;  %in cm: dist. from the rotation center to the detector
 dsd = dso+dod; %in cm: dist. from source to the detector

 
 orbit = 50;     % in degree: angular span
 na = size(g,3); % number of projection views
 ds = 0.0085*2;      % in cm: detector element pixel size in the 's' direction; 
 dt = 0.0085*2;      % in cm: detector element pixel size in the 's' direction; 
                 % 's', the x-ray tube moving direction, positive pointing toward right.           
                 % 't', the perpendicular direction to 's' direction, positive pointing toward the nipple.            
 
 ns       = size(g,1); % number of detector elements in the 's' direction
 nt       = size(g,2); % number of detector elements in the 's' direction
 offset_s = 0;         % detector center offset along the 's' direction in pixels relative to the tube rotation center
 offset_t = -nt/2;     % detector center offset along the 't' direction in pixels relative to the tube rotation center
 d_objbottom_det = 0;  % in cm,the distance from the bottom of the object to the center detector.
%=======================================
%User defines the recon volume geometry:
%=======================================
%%treat the x-ray tube rotation center as the origin of the 3D coordinate
%%system ( see the coordinate system sketch in the instruction document)
%x: posive direction points toward right, 
%y: positive direction points toward the nipple
%z: positive direction points toward the x-ray source
% voxel size (drx, dry, drz) in cm, 
% dimensions (nrx, nry, nrz) and 
% FOV center offsets (offset_x, offset_y, offset_z): in pixels relative to the rotation center.
% For example, if the coordinates of the FOV center is (xctr, yctr, zctr) relative tothe rotation center,
% then offset_x=-xctr, offset_y=-yctr and offset_z=-zctr.
nrx      = 1400;%1880/2;
nry      = 800;%1052/2;
drx      = 0.0085*2;  
dry      = drx; 
drz      = 0.0085*2; 
nrz      = 752/4; 
offset_x = 0;      % in pixels
offset_y = -nry/2; % in pixels. 0 for full cone, -nry/2 for half cone
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
% disp 'FBP'
xfbp = fbp_dbt(Gtr, btg, igr, g, 'hann75');

