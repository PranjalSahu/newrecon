function phantom=readphantom(path, phantomshape)
    fid  = fopen(path, 'r');
    data = fread(fid, phantomshape(1)*phantomshape(2)*phantomshape(3), 'char');  %change the shape of phantom 
    fclose(fid);
    
    phantom = reshape(data, [phantomshape(1), phantomshape(2), phantomshape(3)]);
    phantom = permute(phantom, [2 1 3]);
    phantom = permute(phantom, [1 3 2]);
end


% Voxel size in the phantom is
% 0.2 x 0.2 x 0.2

% show one slice of phantom
% imshow(reshape(phantom(:, 120, :), [329, 939])/10.0);


% Code to get the binary mask of the breast region
% Do it for each slice and then just do the binary multiplcation with the
% volume while back-projecting
%imshow(imerode(imdilate(tp>th,se7),se9));


%fid   = fopen('/home/pranjal/victre/breastPhantom/pc_2011766737_crop.raw', 'r');
%data  = fread(fid, 388*932*419);
%data1 = reshape(data, [388, 932, 419]);
%imshow(data1(:, :, 450)/100.0);


% for t=43:44
%     st = strcat('C:\Users\psahu\TESTBED\projections\', int2str(t), '_250');
%     SART_rename(st);
%     clear; clc
% end