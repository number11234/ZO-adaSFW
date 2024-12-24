function [x_ave_track, eps_track, obj_track,obj_track_50] = ACC_UNS_SenSel(options)
%ZO-ADASFW 此处显示有关此函数的摘要
%   此处显示详细说明
A_allT = options.data;
d = size(A_allT,2); %%% number of optimization variables  50
N = size(A_allT,3); %%% number of random samples   1000
% x = options.x0;  % 改为符合条件的
IterMax = options.IterMax ;   % 迭代数量  
%eps = options.eps ;
ksel = options.ksel;   % 传感器的数量

mu = 1/(d*IterMax^(1/2));      % 平滑参数
B1 = 50;
B2 = 50 ;      % 批次
Q =  10;   % 梯度估计次数   
eta = 1/(IterMax) ;    % 学习率
    % frank 参数
x = ones(d,1)*(ksel/d);
a = x ;
b = x ;



x_track = []; 
x_ave_track = [];  
eps_track = []; 
obj_track = func_global_SenSel(A_allT, x);
old_grad = 0;
obj_track_50 = func_global_SenSel(A_allT, x)/1000;


for t = 1 :IterMax    %% 总迭代次数
    grad = 0;
    ztj_sub_batch = [];
    for j = 1:Q    % Q梯度估计抽样次数
         ztj = rand(size(x))*2-1;
         norm_value = norm(ztj, 2);
         ztj = ztj/norm_value;
         ztj_sub_batch = [ztj_sub_batch,ztj];
    end
    rho = 1/(t)^(2/3);
    if t == 1   %% 收计算满梯度
        idx_sample = randperm(N,B1); % N=1000， 从1-1000里随机选取 Bt个数
        A_sel = A_allT(:,:,idx_sample);   % 将选中的数据提取出来
       
        for i = 1:B1    %% 便利所有的函数,接下来对单个函数怎么求出梯度
            for j = 1:Q
                ztj = ztj_sub_batch(:,j);
                [f_temp1 , grad_temp1] = func_loc_n_SenSel(A_sel(:,:,i),x+mu*ztj);
                [f_temp2 , grad_temp2] = func_loc_n_SenSel(A_sel(:,:,i),x);
                grad = grad + (f_temp1 - f_temp2)/(mu)*ztj*d*(1/(B1*Q));
            end
        end
    else
        idx_sample = randperm(N,B2); % N=1000， 从1-1000里随机选取 Bt个数
        A_sel = A_allT(:,:,idx_sample);   % 将选中的数据提取出来
        grad1 = 0;
        grad2 = 0;
        for i = 1:B2
            for j = 1:Q
                ztj = ztj_sub_batch(:,j);
                [f_temp1 , grad_temp1] = func_loc_n_SenSel(A_sel(:,:,i),x+mu*ztj);
                [f_temp2 , grad_temp2] = func_loc_n_SenSel(A_sel(:,:,i),x);
                grad1 = grad1 + (f_temp1 - f_temp2)/(mu)*d*ztj*(1/(B2*Q));
                [f_temp1_old , grad_temp1] = func_loc_n_SenSel(A_sel(:,:,i),old_x+mu*ztj);
                [f_temp2_old , grad_temp2] = func_loc_n_SenSel(A_sel(:,:,i),old_x);
                grad2 = grad2 + (f_temp1_old - f_temp2_old)/(mu)*ztj*d*(1/(B2*Q));
            end
        end
        grad = grad1 + (1-rho)*(-grad2+old_grad) ;
    end
    gamma = (1+(1/(t+1)/(t+2)))*eta ;
    alpha= 1/(t+1);
    old_x = x ; % 保存旧参数
    old_grad = grad;  % 保存旧梯度
    w = LMO(grad, ksel,d);
    a = a + gamma * (w - a);
    b = x + eta*(w-x);
    x = (1-alpha)*b + alpha*a ;
  
   
    %%% 保存信息
    x_track = [x_track,x];
    x_ave = mean(x_track,2);
    x_ave_track = [x_ave_track,x_ave];
    
    if t > 1
            eps_track = [eps_track; norm(x_ave_track(:,end)-x_ave_track(:,end-1))]; %% 这一行计算最近两次迭代中 x_ave_track 的变化量，并添加到 eps_track 中。
            obj_temp = func_global_SenSel(A_allT, x_ave);  %% 计算目标函数值
            obj_track = [obj_track;  obj_temp];           %% 追踪 目标函数值
            if mod(t,50) == 0    %% 每50打印一次信息
               disp(sprintf('ACC_UNS for iter = %d with xeps = %4.5f, obj = %4.5f',...
                    t, eps_track(end),obj_track(end))); 
                 obj_track_50 = [obj_track_50;  obj_temp/1000];
            end
    end
end


end