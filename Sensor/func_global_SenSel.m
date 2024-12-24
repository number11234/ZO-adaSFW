function obj= func_global_SenSel(A_allT, x)  %% ¼ÆËãº¯ÊýÖµ
    N = size(A_allT,3); obj = 0; 
    for i = 1:N
        an = A_allT(:,:,i);
        A_tmp = an*diag(x)*an.';
        A_tmp = 0.5*(A_tmp + A_tmp.');
        obj = obj + (- log(det(A_tmp)));
    end
end