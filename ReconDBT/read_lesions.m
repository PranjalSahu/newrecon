function [lesions] = read_lesions()

    lesions = zeros([144 144 144 4]);
    
    fin         = fopen('/home/dril/breastMass-master/breastMass/mass_11_107.raw');
    lesion_size = 107;
    I   = fread(fin, lesion_size*lesion_size*lesion_size,'uint8=>uint8');
    Z   = reshape(I, lesion_size, lesion_size, lesion_size);
    Z   = imresize3(Z, 144/lesion_size); 
    Z1   = imbinarize(Z);
    lesions(:, :, :, 1) = Z1;
    
    
    fin         = fopen('/home/dril/breastMass-master/breastMass/mass_13_132.raw');
    lesion_size = 132;
    I   = fread(fin, lesion_size*lesion_size*lesion_size,'uint8=>uint8');
    Z   = reshape(I, lesion_size, lesion_size, lesion_size);
    Z   = imresize3(Z, 144/lesion_size); 
    Z2   = imbinarize(Z);
    lesions(:, :, :, 2) = Z2;
    
    
    fin         = fopen('/home/dril/breastMass-master/breastMass/mass_14_118.raw');
    lesion_size = 118;
    I   = fread(fin, lesion_size*lesion_size*lesion_size,'uint8=>uint8');
    Z   = reshape(I, lesion_size, lesion_size, lesion_size);
    Z   = imresize3(Z, 144/lesion_size); 
    Z3   = imbinarize(Z);
    lesions(:, :, :, 3) = Z3;
    
    fin         = fopen('/home/dril/breastMass-master/breastMass/mass_15_161.raw');
    lesion_size = 161;
    I   = fread(fin, lesion_size*lesion_size*lesion_size,'uint8=>uint8');
    Z   = reshape(I, lesion_size, lesion_size, lesion_size);
    Z   = imresize3(Z, 144/lesion_size); 
    Z4   = imbinarize(Z);
    lesions(:, :, :, 4) = Z4;
end