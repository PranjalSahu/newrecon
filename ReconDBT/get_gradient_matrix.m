%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to get the gradient operator for z direction
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function s = get_gradient_matrix(x)
    t1        = [1:numel(x)];              % get coordinate array
    t2        = t1+size(x, 1);
    t2(t2 > numel(x)) = 1;
    t         = [t1 t2];
    
    v1 = zeros([1, numel(x)])+1;
    v2 = zeros([1, numel(x)])-1;
    v  = [v1 v2];
    
    index_arr = [1:numel(x)];
    index_arr = [index_arr index_arr];
    
    s  = sparse(index_arr, t, v, numel(x), numel(x));
    %x1 = double(reshape(x, [numel(x), 1]));
    %r1 = s*x1;
    %r1 = reshape(r1, size(x));
end