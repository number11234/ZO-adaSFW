%%%%%%%%%%%% Author: Sijia Liu %%%%%%%%%%%%%%%%
%%%%%%%%%%%% Date: 08/02/2017
%%% Application name: ZO-ADMM for sensor selection

clc; clear all; close all;

% clc：清空命令窗口。在这里意味着忽略之前的任何输出。
% clear all：从工作空间中移除所有变量。我们这里没有活动变量，但如果有的话，它们现在会被清除。
% close all：关闭所有图形窗口。既然我们不在 MATLAB 环境中运行，也就没有图形窗口可以关闭，但如果有的话，它们现在会被关闭。



%%% input data: observation gains, location of sensors, and location of
%%% targets/field points to be monitored 
load('dataSample_sensrSel.mat','A_allT','Locs_sensors','Loc_target');
Ntar = size(A_allT,1); %%% num. of field points to be estimated  5     A_allT = 5*50*1000
Nsen = size(A_allT,2); %%% number of sensors, problem size, dimension of optimization variables 参数的维度 50
Ksel_vec = 5:5:20;  %%% desired selected sensors [5 10 15 20]

% 你使用 load 命令加载了三个变量：A_allT（观测增益矩阵）、Locs_sensors（传感器的位置）和
% Loc_target（目标/监测点的位置）.这些数据来自文件 dataSample_sensrSel.mat
% Ntar 被定义为观测增益矩阵 A_allT 的行数，这表示需要估计的目标或监测点的数量。
% Nsen 被定义为观测增益矩阵 A_allT 的列数，这表示传感器的数量，同时也是优化问题的规模。
% Ksel_vec 定义了一个向量，其中包含所需的传感器数量，范围是从 5 到 20，步长为 5。

mse_track_OADMM = zeros(length(Ksel_vec),1);  %%% MSE based on sensor selection schemes using OADMM 。 4行1列的矩阵  
mse_track_ZOADMM = zeros(length(Ksel_vec),1);  %%% MSE based on sensor selection schemes using ZOADMM。 4行1列的矩阵
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



% 创建一个3维数组 Nsen*4*1
% eps_track_OADMM = [];  %%% update error trajectory
% eps_track_ZOADMM = [];



%%%% parameter setting in ZOADMM
d = Nsen; %%Optimization varaibles dimension
options.A = eye(d);    % eye(d) 返回一个d*d大小的单位矩阵
options.B = -eye(d);
options.c = zeros(d,1); %%% coefficients of equality constraint Ax + By = c  创建一个d行1列的零向量。
options.rho = 1;
options.x0 = ones(d,1)*0.5;     % 创建一个d行一列的[0.5][0.5][0.5][0.5]
options.y0 = inv(-options.B)*( options.A * options.x0 - options.c );  %   (Ax-c)* (-B)   inv 是逆矩阵
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
        options.ksel  = Ksel_vec(j);   % 选择向量[5 10 15 20]的数量
        rng(2030);
        
      
        
        
        
%         %% Method 1: second order method, primal dual interior point,  CVX solver should be installed first, http://cvxr.com/cvx/
%         A_tmp = [];
%         for ii = 1:Nsen                     % 参数的维度 50
%             an = squeeze(A_allT(:,ii,:));   % A_allT(5*1*1000) 是一个三维数组。 squeeze（）用于删除数组中的单维或大小为 1 的维度。故an = 5*1000
%             A_tmp = [A_tmp , an*an.'];      % 5*250
%         end
%         
%         T = size(A_allT,3);         % 1000
%         cvx_begin   
%         variable x_cvx(d);          % 设置优化变量 x_cvx 维度为d的列向量
%         minimize -log_det( A_tmp*kron(x_cvx,eye(size(A_allT,1))) ) ;     % 你是在创建一个单位矩阵（identity matrix），其大小等于 A_allT 的第一维的大小。
%         subject to
%             x_cvx>=0;
%             x_cvx<=1;
%             sum(x_cvx) == options.ksel;
%         cvx_end
%         
%        [x_sel_cvx, mse_cvx ]= mse_SenSel(A_allT, x_cvx ,  options.ksel ) ; 
%        % 返回 x_sel_cvx 是一个二进制向量，其中被选中的值设为1. mse_cvx 均方误差
%        
%        xsel_track_cvx(:,j) = x_sel_cvx;  % 追踪 x_cvx 在不同的m0
%        mse_track_cvx(j) = mse_cvx;       % 追踪 mse_cvx 在不同的m0     
%        
%        
%         %%% Method 2: online ADMM
%         options.grad_free = 0; %%% 0: full gradient
%         options.eps = 1e-6;  %%% stopping rule
%         options.L_sub_batch_outter = 50;  options.L_sub_batch_inner = 1; %%% sub-batch strategy  批次策略
%         %%% call algorithm
%         [x_ave_track_OADMM,y_ave_track_OADMM,eps_track_OADMM_tmp, obj_track_OADMM_tmp] = ZOADMM_SenSel(options); %调用函数，后返回变量
%        
%         eps_track_OADMM(:,j) = eps_track_OADMM_tmp; %% 保存不同传感器下的变量
%         obj_track_OADMM(:,j) = obj_track_OADMM_tmp;
%         
%         
%         [x_sel_OADMM, mse_OADMM ]= mse_SenSel(A_allT, x_ave_track_OADMM(:,end) ,  options.ksel ) ; % 计算 mse
%         
%         mse_track_OADMM(j) = mse_OADMM;         %保存mse
%         xsel_track_OADMM(:,j) = x_sel_OADMM;
%         
%         
%       
%         
%         %%% Method 3: ZO-ADMM
%         options.grad_free = 1; options.eps = 1e-6; 
%         options.L_sub_batch_outter = 1;  options.L_sub_batch_inner = 50;
%         
%         [x_ave_track_ZOADMM,y_ave_track_ZOADMM,eps_track_ZOADMM_tmp, obj_track_ZOADMM_tmp] = ZOADMM_SenSel(options);  % 调用算法
%         
%         eps_track_ZOADMM(:,j) = eps_track_ZOADMM_tmp; % 保存 
%         obj_track_ZOADMM(:,j) = obj_track_ZOADMM_tmp;
%         
%         [x_sel_ZOADMM, mse_ZOADMM ]= mse_SenSel(A_allT, x_ave_track_ZOADMM(:,end) ,  options.ksel ) ;  % 计算
%         mse_track_ZOADMM(j) = mse_ZOADMM;
%         xsel_track_ZOADMM(:,j) = x_sel_ZOADMM;
% %         
        
         % Method1：zhu
         [x_ave_track_ZO_adaSFW, eps_track_ZO_adaSFW_tmp, obj_track_ZO_adaSFW_tmp] = ZO_adaSFW_SenSel(options);
         eps_track_ZO_adaSFW(:,j) = eps_track_ZO_adaSFW_tmp; 
         obj_track_ZO_adaSFW(:,j) = obj_track_ZO_adaSFW_tmp;
         
         [x_sel_ZO_adaSFW, mse_ZO_adaSFW ]= mse_SenSel(A_allT, x_ave_track_ZO_adaSFW(:,end) ,  options.ksel ) ; 
         
         mse_track_ZO_adaSFW(j) = mse_ZO_adaSFW;         
         xsel_track_ZO_adaSFW(:,j) = x_sel_ZO_adaSFW;
        
       
        % Method2: SFW_grad
        [x_ave_track_SFW_grad, eps_track_SFW_grad_tmp, obj_track_SFW_grad_tmp, obj_track_SFW_grad_50_tmp] = SFW_grad_SenSel(options);
        eps_track_SFW_grad(:,j) = eps_track_SFW_grad_tmp; %% 保存不同传感器下的变量
        obj_track_SFW_grad(:,j) = obj_track_SFW_grad_tmp;
        obj_track_SFW_grad_50(:,j) = obj_track_SFW_grad_50_tmp;
        
        [x_sel_SFW_grad, mse_SFW_grad ]= mse_SenSel(A_allT, x_ave_track_SFW_grad(:,end) ,  options.ksel ) ; % 计算 mse
        
        mse_track_SFW_grad(j) = mse_SFW_grad;         %保存mse
        xsel_track_SFW_grad(:,j) = x_sel_SFW_grad;
        
       
%             %% Method5:ZSCG
%         [x_ave_track_ZSCG, eps_track_ZSCG_tmp, obj_track_ZSCG_tmp,obj_track_ZSCG_50_tmp] = ZSCG_SenSel(options);
%         eps_track_ZSCG(:,j) = eps_track_ZSCG_tmp; %% 保存不同传感器下的变量
%         obj_track_ZSCG(:,j) = obj_track_ZSCG_tmp;
%         obj_track_ZSCG_50(:,j) = obj_track_ZSCG_50_tmp;
%         
%         [x_sel_ZSCG, mse_ZSCG ]= mse_SenSel(A_allT, x_ave_track_ZSCG(:,end) ,  options.ksel ) ; % 计算 mse
%         
%         mse_track_ZSCG(j) = mse_ZSCG;         %保存mse
%         xsel_track_ZSCG(:,j) = x_sel_ZSCG;
%        
%    
% 
%         
%      
%         
%        %% Method7:FZFW
%         [x_ave_track_FZFW, eps_track_FZFW_tmp, obj_track_FZFW_tmp] = FZFW_SenSel(options);
%         eps_track_FZFW(:,j) = eps_track_FZFW_tmp; %% 保存不同传感器下的变量
%         obj_track_FZFW(:,j) = obj_track_FZFW_tmp;
%         
%         [x_sel_FZFW, mse_FZFW ]= mse_SenSel(A_allT, x_ave_track_FZFW(:,end) ,  options.ksel ) ; % 计算 mse
%         
%         mse_track_FZFW(j) = mse_FZFW;         %保存mse
%         xsel_track_FZFW(:,j) = x_sel_FZFW;
        
         %% Method6:ZO-SFW
        [x_ave_track_ZO_SFW, eps_track_ZO_SFW_tmp, obj_track_ZO_SFW_tmp, obj_track_ZO_SFW_50_tmp] = ZO_SFW_SenSel(options);
        eps_track_ZO_SFW(:,j) = eps_track_ZO_SFW_tmp; %% 保存不同传感器下的变量
        obj_track_ZO_SFW(:,j) = obj_track_ZO_SFW_tmp;
        obj_track_ZO_SFW_50(:,j) = obj_track_ZO_SFW_50_tmp;
        
        [x_sel_ZO_SFW, mse_ZO_SFW ]= mse_SenSel(A_allT, x_ave_track_ZO_SFW(:,end) ,  options.ksel ) ; % 计算 mse
        
        mse_track_ZO_SFW(j) = mse_ZO_SFW;         %保存mse
        xsel_track_ZO_SFW(:,j) = x_sel_ZO_SFW;
      
       
        
      %% Method9:ACC_UN
%         [x_ave_track_ACC_UN, eps_track_ACC_UN_tmp, obj_track_ACC_UN_tmp,obj_track_ACC_UN_50_tmp] = ACC_UN_SenSel(options);
%         eps_track_ACC_UN(:,j) = eps_track_ACC_UN_tmp; %% 保存不同传感器下的变量
%         obj_track_ACC_UN(:,j) = obj_track_ACC_UN_tmp;
%         obj_track_ACC_UN_50(:,j) = obj_track_ACC_UN_50_tmp;
%         
%         [x_sel_ACC_UN, mse_ACC_UN ]= mse_SenSel(A_allT, x_ave_track_ACC_UN(:,end) ,  options.ksel ) ; % 计算 mse
%         
%         mse_track_ACC_UN(j) = mse_ACC_UN;         %保存mse
%         xsel_track_ACC_UN(:,j) = x_sel_ACC_UN; 
%         
%         
%         % Method10:ACC_UNS
%         [x_ave_track_ACC_UNS, eps_track_ACC_UNS_tmp, obj_track_ACC_UNS_tmp,obj_track_ACC_UNS_50_tmp] = ACC_UNS_SenSel(options);
%         eps_track_ACC_UNS(:,j) = eps_track_ACC_UNS_tmp; %% 保存不同传感器下的变量
%         obj_track_ACC_UNS(:,j) = obj_track_ACC_UNS_tmp;
%         obj_track_ACC_UNS_50(:,j) = obj_track_ACC_UNS_50_tmp;
%         
%         [x_sel_ACC_UNS, mse_ACC_UNS ]= mse_SenSel(A_allT, x_ave_track_ACC_UNS(:,end) ,  options.ksel ) ; % 计算 mse
%         
%         mse_track_ACC_UNS(j) = mse_ACC_UNS;         %保存mse
%         xsel_track_ACC_UNS(:,j) = x_sel_ACC_UNS; 
%         
%         %% Method11:ACC_CO
%         [x_ave_track_ACC_CO, eps_track_ACC_CO_tmp, obj_track_ACC_CO_tmp] = ACC_Co_SenSel(options);
%         eps_track_ACC_CO(:,j) = eps_track_ACC_CO_tmp; %% 保存不同传感器下的变量
%         obj_track_ACC_CO(:,j) = obj_track_ACC_CO_tmp;
%         
%         [x_sel_ACC_CO, mse_ACC_CO ]= mse_SenSel(A_allT, x_ave_track_ACC_CO(:,end) ,  options.ksel ) ; % 计算 mse
%         
%         mse_track_ACC_CO(j) = mse_ACC_CO;         %保存mse
%         xsel_track_ACC_CO(:,j) = x_sel_ACC_CO; 
%         
%         %% Method12:ACC_COS
%         [x_ave_track_ACC_COS, eps_track_ACC_COS_tmp, obj_track_ACC_COS_tmp] = ACC_COS_SenSel(options);
%         eps_track_ACC_COS(:,j) = eps_track_ACC_COS_tmp; %% 保存不同传感器下的变量
%         obj_track_ACC_COS(:,j) = obj_track_ACC_COS_tmp;
%         
%         [x_sel_ACC_COS, mse_ACC_COS ]= mse_SenSel(A_allT, x_ave_track_ACC_COS(:,end) ,  options.ksel ) ; % 计算 mse
%         
%         mse_track_ACC_COS(j) = mse_ACC_COS;         %保存mse
%         xsel_track_ACC_COS(:,j) = x_sel_ACC_COS; 
%         
% %         
%          %% Method13:Ada_SFW
%         [x_ave_track_Ada_SFW, eps_track_Ada_SFW_tmp, obj_track_Ada_SFW_tmp,obj_track_Ada_SFW_50_tmp] = AdaSFW_SenSel(options);
%         eps_track_Ada_SFW(:,j) = eps_track_Ada_SFW_tmp; %% 保存不同传感器下的变量
%         obj_track_Ada_SFW(:,j) = obj_track_Ada_SFW_tmp;
%         obj_track_Ada_SFW_50(:,j) = obj_track_Ada_SFW_50_tmp;
%         
%         [x_sel_Ada_SFW, mse_Ada_SFW ]= mse_SenSel(A_allT, x_ave_track_Ada_SFW(:,end) ,  options.ksel ) ; % 计算 mse
%         
%         mse_track_Ada_SFW(j) = mse_Ada_SFW;         %保存mse
%         xsel_track_Ada_SFW(:,j) = x_sel_Ada_SFW;
% %   

%        disp(sprintf('adaSFW for mse = %4.10f',...
%            mse_ZO_adaSFW )); 
%        disp(sprintf('mse_ZSCG for mse = %4.10f',...
%            mse_ZSCG )); 
%        disp(sprintf('mse_ZO_SFW for mse = %4.10f',...
%            mse_ZO_SFW )); 
%        disp(sprintf('mse_FZFW for mse = %4.10f',...
%            mse_FZFW )); 
%        disp(sprintf('mse_SFW_grad for mse = %4.10f',...
%            mse_SFW_grad )); 
%        disp(sprintf('mse_ACC_UN for mse = %4.10f',...
%            mse_ACC_UN )); 
%        disp(sprintf('mse_ACC_UNS for mse = %4.10f',...
%            mse_ACC_UNS )); 
%        disp(sprintf('mse_ACC_CO for mse = %4.10f',...
%            mse_ACC_CO )); 
%        disp(sprintf('mse_ACC_COS for mse = %4.10f',...
%            mse_ACC_COS )); 
%        disp(sprintf('mse_Ada_SFW for mse = %4.10f',...
%            mse_Ada_SFW )); 

end

save('mse_data.mat','mse_track_ZO_adaSFW','mse_track_ZSCG','mse_track_ZO_SFW','mse_track_ACC_UN','mse_track_ACC_CO','mse_track_ACC_UNS','mse_track_ACC_COS','mse_track_FZFW','mse_track_SFW_grad','mse_track_Ada_SFW');
%save('obj_data.mat','obj_track_OADMM','obj_track_cvx','obj_track_ZOADMM','obj_track_ZO_adaSFW','obj_track_ZSCG','obj_track_OADMM');

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
legend('CVX','O-ADMM','ZO-ADMM','ZO-adaSFW')

hfig = figure;
Ksel_vec = 5:5:20;
ZSCG = [0.000801953842497996,0.000401053950729963,0.000267258841768363,0.000199955071027114];
ZO_SFW = [0.000803327898867393,0.000401752280888226 0.00026670744690928,0.000199222747193904];
ACC_SZOFW_UniGE = [0.000797566007232073,0.000399582446583582,0.00026632036685417,0.000199680055492744];
ACC_SZOFW_CooGE = [0.000803327898867393,0.000399030063869504,0.000266171175575597,0.00019952799985446 ];
ACC_SZOFW_UniGEs = [0.000803327898867393,0.000402051942866367,0.000265825743713173,0.000199230151001602];
ACC_SZOFW_CooGEs = [0.000803327898867393,0.000399618611514102,0.000265606067918843,0.000199206536876526];
FZFW = [0.000803327898867393,0.000399618611514102, 0.00026614601184508, 0.000199206536876526];
SFW_Grad = [0.000805607334690796,0.000401053950729963,0.000267290603638329,0.000199955071027114];
AdaSFW = [0.000803327898867393, 0.000399618611514102, 0.000265606067918843, 0.000199206536876526];
ZO_adaSFW = [0.000803327898867393,0.000399618611514102,0.000265606067918843,0.000199206536876526  ];

plot(Ksel_vec,ZSCG,'-ro'); hold on;
plot(Ksel_vec,ZO_SFW,'-b^'); hold on;
plot(Ksel_vec,ACC_SZOFW_UniGE,'-ks'); hold on
plot(Ksel_vec,ACC_SZOFW_CooGE,'-gx');hold on
plot(Ksel_vec,ACC_SZOFW_UniGEs,'-cs'); hold on
plot(Ksel_vec,ACC_SZOFW_CooGEs,'-mx');hold on
plot(Ksel_vec,FZFW,'-yd'); hold on;
plot(Ksel_vec,SFW_Grad,'-k+'); hold on;
plot(Ksel_vec,AdaSFW,'-go'); hold on;
plot(Ksel_vec,ZO_adaSFW,'-ko'); 
xlabel('No. of selected sensors')
ylabel('Mean sqaured error');
legend('CVX','O-ADMM','ZO-ADMM','ZO-adaSFW')




