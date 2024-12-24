

clc; clear all; close all;
 
load('dataSample_sensrSel.mat','A_allT','Locs_sensors','Loc_target');
Ntar = size(A_allT,1); %%% num. of field points to be estimated  5     
Nsen = size(A_allT,2); %%% number of sensors, problem size, dimension of optimization variables 
Ksel_vec = 5:5:20;  %%% desired selected sensors 


mse_track_OADMM = zeros(length(Ksel_vec),1);  %%% MSE based on sensor selection schemes using OADMM ¡£   
mse_track_ZOADMM = zeros(length(Ksel_vec),1);  %%% MSE based on sensor selection schemes using ZOADMM¡£
mse_track_ZO_adaSFW = zeros(length(Ksel_vec),1);
mse_track_ZSCG = zeros(length(Ksel_vec),1);
mse_track_ZO_SFW = zeros(length(Ksel_vec),1);
mse_track_ACC_UN = zeros(length(Ksel_vec),1);
mse_track_ACC_CO = zeros(length(Ksel_vec),1);
mse_track_ACC_UNS = zeros(length(Ksel_vec),1);
mse_track_ACC_COS = zeros(length(Ksel_vec),1);
mse_track_FZFW = zeros(length(Ksel_vec),1);
mse_track_SFW_grad = zeros(length(Ksel_vec),1);
mse_track_Ada_SFW = zeros(length(Ksel_vec),1);


xsel_track_OADMM = zeros(Nsen,length(Ksel_vec),1);
xsel_track_ZOADMM = zeros(Nsen,length(Ksel_vec),1);
xsel_track_ZO_adaSFW = zeros(Nsen,length(Ksel_vec),1);
xsel_track_ZSCG = zeros(Nsen,length(Ksel_vec),1);
xsel_track_ZO_SFW = zeros(Nsen,length(Ksel_vec),1);






%%%% parameter setting in ZOADMM
d = Nsen; %%Optimization varaibles dimension
options.A = eye(d);    
options.B = -eye(d);
options.c = zeros(d,1); %%% coefficients of equality constraint Ax + By = c ¡£
options.rho = 1;
options.x0 = ones(d,1)*0.5;     
options.y0 = inv(-options.B)*( options.A * options.x0 - options.c );  
options.lam0 = zeros(d,1);
options.a = 10; %%% Gt = aI - \eta_t*rho A'*A, Bregman divergence coefficient matrix
options.eta_const = 1; %%% related to step size for Bregman divergence, learning rate
options.grad_est_const = 1; %%% related to step size of directional derivative, smoothing parameter
options.IterMax = 2000; %%% max. iteration
options.data = A_allT;  %%% online data
%%%% ensure Gt >= I   


%%% parameter setting in ZO-adaSFW
options.q = 2;
options.mu = 1 / (sqrt(d) * sqrt(2e4));
options.Bt = 50;
options.ada_eps = 1e-8;
options.K = 5;
options.eta = 1;
options.base_gamma = 1;



while 1
        if min(  eig( options.a*eye(d) - (options.eta_const)*options.rho *options.A.'*options.A ) ) < 1
            options.a = options.a*1.1;
        else
            break;
        end
end

for j = 1:length(Ksel_vec) %%% different numbers of selected sensors  [5 10 15 20]. j=1,2,3,4
        options.ksel  = Ksel_vec(j);   
        rng(2030);
        
      
        
        
        
%         %% Method 1: second order method, primal dual interior point,  CVX solver should be installed first, http://cvxr.com/cvx/
%         A_tmp = [];
%         for ii = 1:Nsen                     
%             an = squeeze(A_allT(:,ii,:));   
%             A_tmp = [A_tmp , an*an.'];     
%         end
%         
%         T = size(A_allT,3);       
%         cvx_begin   
%         variable x_cvx(d);          
%         minimize -log_det( A_tmp*kron(x_cvx,eye(size(A_allT,1))) ) ;    
%         subject to
%             x_cvx>=0;
%             x_cvx<=1;
%             sum(x_cvx) == options.ksel;
%         cvx_end
%         
%        [x_sel_cvx, mse_cvx ]= mse_SenSel(A_allT, x_cvx ,  options.ksel ) ; 
%       
%        
%        xsel_track_cvx(:,j) = x_sel_cvx;  
%        mse_track_cvx(j) = mse_cvx;           
%        
%        
%         %%% Method 2: online ADMM
%         options.grad_free = 0; %%% 0: full gradient
%         options.eps = 1e-6;  %%% stopping rule
%         options.L_sub_batch_outter = 50;  options.L_sub_batch_inner = 1; %%% sub-batch strategy  
%         %%% call algorithm
%         [x_ave_track_OADMM,y_ave_track_OADMM,eps_track_OADMM_tmp, obj_track_OADMM_tmp] = ZOADMM_SenSel(options); 
%        
%         eps_track_OADMM(:,j) = eps_track_OADMM_tmp; 
%         obj_track_OADMM(:,j) = obj_track_OADMM_tmp;
%         
%         
%         [x_sel_OADMM, mse_OADMM ]= mse_SenSel(A_allT, x_ave_track_OADMM(:,end) ,  options.ksel ) ; 
%         
%         mse_track_OADMM(j) = mse_OADMM;         
%         xsel_track_OADMM(:,j) = x_sel_OADMM;
%         
%         
%       
%         
%         %%% Method 3: ZO-ADMM
%         options.grad_free = 1; options.eps = 1e-6; 
%         options.L_sub_batch_outter = 1;  options.L_sub_batch_inner = 50;
%         
%         [x_ave_track_ZOADMM,y_ave_track_ZOADMM,eps_track_ZOADMM_tmp, obj_track_ZOADMM_tmp] = ZOADMM_SenSel(options); 
%         
%         eps_track_ZOADMM(:,j) = eps_track_ZOADMM_tmp;  
%         obj_track_ZOADMM(:,j) = obj_track_ZOADMM_tmp;
%         
%         [x_sel_ZOADMM, mse_ZOADMM ]= mse_SenSel(A_allT, x_ave_track_ZOADMM(:,end) ,  options.ksel ) ; 
%         mse_track_ZOADMM(j) = mse_ZOADMM;
%         xsel_track_ZOADMM(:,j) = x_sel_ZOADMM;
% %         
       
         % Method1£ºzhu
        [x_ave_track_ZO_adaSFW, eps_track_ZO_adaSFW_tmp, obj_track_ZO_adaSFW_tmp] = ZO_adaSFW_SenSel(options);
        eps_track_ZO_adaSFW(:,j) = eps_track_ZO_adaSFW_tmp; 
        obj_track_ZO_adaSFW(:,j) = obj_track_ZO_adaSFW_tmp;
        
        [x_sel_ZO_adaSFW, mse_ZO_adaSFW ]= mse_SenSel(A_allT, x_ave_track_ZO_adaSFW(:,end) ,  options.ksel ) ; 
        
        mse_track_ZO_adaSFW(j) = mse_ZO_adaSFW;         
        xsel_track_ZO_adaSFW(:,j) = x_sel_ZO_adaSFW;
       
       
         %% Method2:SFW_grad
        [x_ave_track_SFW_grad, eps_track_SFW_grad_tmp, obj_track_SFW_grad_tmp, obj_track_SFW_grad_50_tmp] = SFW_grad_SenSel(options);
        eps_track_SFW_grad(:,j) = eps_track_SFW_grad_tmp; 
        obj_track_SFW_grad(:,j) = obj_track_SFW_grad_tmp;
        obj_track_SFW_grad_50(:,j) = obj_track_SFW_grad_50_tmp;
        
        [x_sel_SFW_grad, mse_SFW_grad ]= mse_SenSel(A_allT, x_ave_track_SFW_grad(:,end) ,  options.ksel ) ; 
        
        mse_track_SFW_grad(j) = mse_SFW_grad;         
        xsel_track_SFW_grad(:,j) = x_sel_SFW_grad;
        
       
          %% Method3:ZSCG
        [x_ave_track_ZSCG, eps_track_ZSCG_tmp, obj_track_ZSCG_tmp,obj_track_ZSCG_50_tmp] = ZSCG_SenSel(options);
        eps_track_ZSCG(:,j) = eps_track_ZSCG_tmp; 
        obj_track_ZSCG(:,j) = obj_track_ZSCG_tmp;
        obj_track_ZSCG_50(:,j) = obj_track_ZSCG_50_tmp;
        
        [x_sel_ZSCG, mse_ZSCG ]= mse_SenSel(A_allT, x_ave_track_ZSCG(:,end) ,  options.ksel ) ; 
        
        mse_track_ZSCG(j) = mse_ZSCG;         
        xsel_track_ZSCG(:,j) = x_sel_ZSCG;
       
   

        
     
        
       %% Method4:FZFW
        [x_ave_track_FZFW, eps_track_FZFW_tmp, obj_track_FZFW_tmp] = FZFW_SenSel(options);
        eps_track_FZFW(:,j) = eps_track_FZFW_tmp; 
        obj_track_FZFW(:,j) = obj_track_FZFW_tmp;
        
        [x_sel_FZFW, mse_FZFW ]= mse_SenSel(A_allT, x_ave_track_FZFW(:,end) ,  options.ksel ) 
        
        mse_track_FZFW(j) = mse_FZFW;         
        xsel_track_FZFW(:,j) = x_sel_FZFW;
        
         % Method5:ZO-SFW
        [x_ave_track_ZO_SFW, eps_track_ZO_SFW_tmp, obj_track_ZO_SFW_tmp, obj_track_ZO_SFW_50_tmp] = ZO_SFW_SenSel(options);
        eps_track_ZO_SFW(:,j) = eps_track_ZO_SFW_tmp; 
        obj_track_ZO_SFW(:,j) = obj_track_ZO_SFW_tmp;
        obj_track_ZO_SFW_50(:,j) = obj_track_ZO_SFW_50_tmp;
        
        [x_sel_ZO_SFW, mse_ZO_SFW ]= mse_SenSel(A_allT, x_ave_track_ZO_SFW(:,end) ,  options.ksel ) ; 
        
        mse_track_ZO_SFW(j) = mse_ZO_SFW;         
        xsel_track_ZO_SFW(:,j) = x_sel_ZO_SFW;
      
       
        
      % Method6:ACC_UN
        [x_ave_track_ACC_UN, eps_track_ACC_UN_tmp, obj_track_ACC_UN_tmp,obj_track_ACC_UN_50_tmp] = ACC_UN_SenSel(options);
        eps_track_ACC_UN(:,j) = eps_track_ACC_UN_tmp;
        obj_track_ACC_UN(:,j) = obj_track_ACC_UN_tmp;
        obj_track_ACC_UN_50(:,j) = obj_track_ACC_UN_50_tmp;
        
        [x_sel_ACC_UN, mse_ACC_UN ]= mse_SenSel(A_allT, x_ave_track_ACC_UN(:,end) ,  options.ksel ) ;
        
        mse_track_ACC_UN(j) = mse_ACC_UN;        
        xsel_track_ACC_UN(:,j) = x_sel_ACC_UN; 
        
        
        % Method7:ACC_UNS
        [x_ave_track_ACC_UNS, eps_track_ACC_UNS_tmp, obj_track_ACC_UNS_tmp,obj_track_ACC_UNS_50_tmp] = ACC_UNS_SenSel(options);
        eps_track_ACC_UNS(:,j) = eps_track_ACC_UNS_tmp; 
        obj_track_ACC_UNS(:,j) = obj_track_ACC_UNS_tmp;
        obj_track_ACC_UNS_50(:,j) = obj_track_ACC_UNS_50_tmp;
        
        [x_sel_ACC_UNS, mse_ACC_UNS ]= mse_SenSel(A_allT, x_ave_track_ACC_UNS(:,end) ,  options.ksel ) ; 
        
        mse_track_ACC_UNS(j) = mse_ACC_UNS;         
        xsel_track_ACC_UNS(:,j) = x_sel_ACC_UNS; 
        
        %% Method8:ACC_CO
        [x_ave_track_ACC_CO, eps_track_ACC_CO_tmp, obj_track_ACC_CO_tmp] = ACC_Co_SenSel(options);
        eps_track_ACC_CO(:,j) = eps_track_ACC_CO_tmp; 
        obj_track_ACC_CO(:,j) = obj_track_ACC_CO_tmp;
        
        [x_sel_ACC_CO, mse_ACC_CO ]= mse_SenSel(A_allT, x_ave_track_ACC_CO(:,end) ,  options.ksel ) ; % 
        
        mse_track_ACC_CO(j) = mse_ACC_CO;         
        xsel_track_ACC_CO(:,j) = x_sel_ACC_CO; 
        
        %% Method9:ACC_COS
        [x_ave_track_ACC_COS, eps_track_ACC_COS_tmp, obj_track_ACC_COS_tmp] = ACC_COS_SenSel(options);
        eps_track_ACC_COS(:,j) = eps_track_ACC_COS_tmp; 
        obj_track_ACC_COS(:,j) = obj_track_ACC_COS_tmp;
        
        [x_sel_ACC_COS, mse_ACC_COS ]= mse_SenSel(A_allT, x_ave_track_ACC_COS(:,end) ,  options.ksel ) ;
        
        mse_track_ACC_COS(j) = mse_ACC_COS;         
        xsel_track_ACC_COS(:,j) = x_sel_ACC_COS; 
        
%         
         %% Method10:Ada_SFW
        [x_ave_track_Ada_SFW, eps_track_Ada_SFW_tmp, obj_track_Ada_SFW_tmp,obj_track_Ada_SFW_50_tmp] = AdaSFW_SenSel(options);
        eps_track_Ada_SFW(:,j) = eps_track_Ada_SFW_tmp; 
        obj_track_Ada_SFW(:,j) = obj_track_Ada_SFW_tmp;
        obj_track_Ada_SFW_50(:,j) = obj_track_Ada_SFW_50_tmp;
        
        [x_sel_Ada_SFW, mse_Ada_SFW ]= mse_SenSel(A_allT, x_ave_track_Ada_SFW(:,end) ,  options.ksel ) ; 
        
        mse_track_Ada_SFW(j) = mse_Ada_SFW;         
        xsel_track_Ada_SFW(:,j) = x_sel_Ada_SFW;
%   

       disp(sprintf('adaSFW for mse = %4.10f',...
           mse_ZO_adaSFW )); 
       disp(sprintf('mse_ZSCG for mse = %4.10f',...
           mse_ZSCG )); 
       disp(sprintf('mse_ZO_SFW for mse = %4.10f',...
           mse_ZO_SFW )); 
       disp(sprintf('mse_FZFW for mse = %4.10f',...
           mse_FZFW )); 
       disp(sprintf('mse_SFW_grad for mse = %4.10f',...
           mse_SFW_grad )); 
       disp(sprintf('mse_ACC_UN for mse = %4.10f',...
           mse_ACC_UN )); 
       disp(sprintf('mse_ACC_UNS for mse = %4.10f',...
           mse_ACC_UNS )); 
       disp(sprintf('mse_ACC_CO for mse = %4.10f',...
           mse_ACC_CO )); 
       disp(sprintf('mse_ACC_COS for mse = %4.10f',...
           mse_ACC_COS )); 
       disp(sprintf('mse_Ada_SFW for mse = %4.10f',...
           mse_Ada_SFW )); 

end

save('mse_data.mat','mse_track_ZO_adaSFW','mse_track_ZSCG','mse_track_ZO_SFW','mse_track_ACC_UN','mse_track_ACC_CO','mse_track_ACC_UNS','mse_track_ACC_COS','mse_track_FZFW','mse_track_SFW_grad','mse_track_Ada_SFW');


%%% MSE
hfig = figure;
% 
plot(Ksel_vec,mse_track_ZSCG,'-ro'); hold on;
plot(Ksel_vec,mse_track_ZO_SFW,'-b^'); hold on;
plot(Ksel_vec, mse_track_ACC_UN,'-ks'); hold on
plot(Ksel_vec,mse_track_ACC_CO,'-gx');hold on
plot(Ksel_vec,mse_track_ACC_UNS,'-cs'); hold on
plot(Ksel_vec,mse_track_ACC_COS,'-mx');hold on
plot(Ksel_vec, mse_track_FZFW,'-yd'); hold on;
plot(Ksel_vec,mse_track_SFW_grad,'-k+'); hold on;
plot(Ksel_vec,mse_track_Ada_SFW,'-go'); hold on;
plot(Ksel_vec,mse_track_ZO_adaSFW,'-ko'); 
xlabel('No. of selected sensors')
ylabel('Mean sqaured error');
legend('ZSCG','ZO_SFW','ACC_UN','ACC_CO','ACC_UNS','ACC_COS','FZFW','SFW_grad','Ada_SFW','ZO_adaSFW')






