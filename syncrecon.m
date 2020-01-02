%he = fopen('/media/pranjal/newdrive/PRANJAL/OSTR/OSTR_SBU/DATA/CE16_HE/host_est_4_0.6_1000000.raw', 'r');
%he = fopen('/media/pranjal/newdrive/PRANJAL/OSTR/OSTR_SBU/DATA/CE16_HE/host_est_4_0.03_20000.raw', 'r');

he = fopen('/media/pranjal/newdrive/HHuang/BR3D/OSTR_HE/host_est_4_0.01_20000.raw', 'r');
he = fread(he, 2000*1000*48, 'float');
he = reshape(he, 2000, 1000, 48);
t  = graythresh(he);
t1 = imbinarize(he, t);


%le = fopen('/media/pranjal/newdrive/PRANJAL/OSTR/OSTR_SBU/DATA/CE16_LE/host_est_4_0.03_1000000.raw', 'r');
%le = fopen('/media/pranjal/newdrive/PRANJAL/OSTR/OSTR_SBU/DATA/CE16_LE/host_est_4_0.03_20000.raw', 'r');

le = fopen('/media/pranjal/newdrive/HHuang/BR3D/OSTR_LE/host_est_4_0.01_20000.raw', 'r');
le = fread(le, 2000*1000*48, 'float');
le = reshape(le, 2000, 1000, 48);


%he = he.*t1;
%le = le.*t1;

%weights = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09];
weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
%weights = [0.50, 0.51, 0.52, 0.53];

for weight = weights
%weight = 0.33;

d = he - le*weight;
%d(d < 0) = 0;

%d = d-min(d, [], 'all');
%d = d.*t1;
histogram(reshape(d, [1, numel(d)]))

disp(weight);
disp(entropy(d(:, :, 24)));

fout = fopen(strcat(['/media/pranjal/newdrive/HHuang/BR3D/OSTR_LE/d_', num2str(weight),'.raw']), 'w');
fwrite(fout, d, 'float');
fclose(fout);
end