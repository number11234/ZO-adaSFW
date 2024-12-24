function [f , grad]= func_loc_n_SenSel(an, x)  % 输入选中的数据，变量x
    Atmp = an*diag(x)*an.';
    Atmp = 0.5*(Atmp + Atmp.');
    f = - log(det(Atmp));
    Ainv_temp = inv(Atmp);
    grad =  -diag(an.'*Ainv_temp*an);
end