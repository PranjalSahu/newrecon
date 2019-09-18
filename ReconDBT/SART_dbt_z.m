function [img,cost,diff_image_final] = SART_dbt_z(Gt,proj,x0,deep,niter,stepsize,saveiter)
% function [img, cost] = SART_dbt(Gt,proj,x0,niter,stepsize,saveiter)
% SART reconstruction for DBT.
% Inputs:
%   Gt: DBT system projector operator, created using "Gtomo_syn()".
%   proj: DBT projection views in the form of line integral, a 3D array of 
%       size ns x nt x na, where ns and nt are the projection image dimensions
%       and na is the number of views.
%   x0: an initial estimate of the reconstuction volume.
%   niter: number of iterations for SART.
%   stepsize: the stepsize for each update.
%   saveiter: 0 or 1. if "0",only output the reconstruction of the final
%           iteration; if "1", output the reconstruction at all iterations.
%
%Outputs:
%   img: the SART reconstructed volume. 
%       If "saveiter" is 0, then "img" is a 3D array contains the volume 
%       of the final iternation;
%       If "saveiter" is 1, then "img" is 1 4D array contains the volumes 
%       of all the iterations.
%   cost: a niterx1 vector containing the cost function value at each iteration
%

if(nargin<7)
    saveiter=0;
end
nview =length(Gt);
arg=Gt{1}.arg;
nx=arg.ig.nx;
ny=arg.ig.ny;
nz=arg.ig.nz;
dz=arg.ig.dz;
nmask=sum(arg.ig.mask(:));

ns=arg.cg.ns;%arg.nn(1);
nt=arg.cg.nt;%arg.nn(2);
nd=arg.nd;

if(saveiter)
    img=zeros([size(x0) niter]);
else
    img=zeros([size(x0) 1]);
end


diff_image_final = zeros(size(proj, 1), size(proj, 2), 25);


y    = zeros(ns,nt);
x    = permute(x0, [1 3 2]);

% For taking the gradient along the z direction
deep = double(permute(deep, [1 3 2]));
deep = reshape(deep, [numel(deep), 1]);

% get the gradient operator which will be used to reduce the
% z direction blurring
S = get_gradient_matrix(x);

for iter=1:niter
    iter
    for i=1:nview
        
        G = Gt{i};

        l     = sum(G');
        l     = reshape(l,ns,nt);
        lmask = logical(l>5*dz);
         
         %Mask out too small l value to avoid blowup in y:
         %some l can be very small. When it is divided from p, 
         %the y value can be overamplified at the boundary(the 
         %boundary value could be 10 or 20 times larger than the
         %normal object value.
         
         pdif     = proj(:,:,i)-G*x;
         y(lmask) = pdif(lmask)./l(lmask);
    
         imgi   = G'*y(:);
         %size(imgi)
         
         denomi = sum(G);
         imgj   = imgi./denomi(:);
         imgj(isnan(imgj)) = 0;
         
         x1         = double(reshape(x, [numel(x), 1]));
         gradz_diff = S*deep - S*x1;
         disp('max is ');
         disp(max(abs(gradz_diff), [], 'all'));
         
         
         gradz_vol  = S'*gradz_diff;
         gradz_vol(gradz_vol > 1) = 1;
         
         %disp('mean is ');
         %disp(mean(gradz_diff, 'all'));
         
         lambda_val = 0.5;
         
         x = x + stepsize*((1-lambda_val)*reshape(imgj, size(x)) + lambda_val*reshape(gradz_vol, size(x)));
         %x = x + stepsize*(1-lambda_val)*reshape(imgj, size(x));
    end
    
    
    
    
    x(x<0)=0;
    %calculate the cost
    if(nargout>1)
        mse=0;
        for i=1:nview
            pdif   = proj(:,:,i)-Gt{i}*x;
            mse    = mse + sum(pdif(:).^2);
            diff_image_final(:, :, i) = pdif;
        end
        cost(iter)=mse;
    end
    
    if (saveiter)
        img(:,:,:,iter) = permute(x,[1 3 2]);
    end
end

%img=embed(img./denom(:),arg.mask);
%img=embed(x,arg.mask);

if(~saveiter)
    img = permute(x,[1 3 2]);
end
