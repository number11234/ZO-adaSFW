function [x_ave_track, eps_track, obj_track] = FZFW_SenSel(options)
%ZO-ADASFW 此处显示有关此函数的摘要
%   此处显示详细说明
A_allT = options.data;
d = size(A_allT,2); %%% number of optimization variables  50
N = size(A_allT,3); %%% number of random samples   1000
% x = options.x0;  % 改为符合条件的
IterMax = options.IterMax ;   % 迭代数量  
%eps = options.eps ;
ksel = options.ksel;   % 传感器的数量
q = 2;         % 更换梯度
mu = 1/(sqrt(IterMax*d)) ;      % 平滑参数
lr = 1/(d*sqrt(IterMax));
B1 = 50;
B2 = 50 ;      % 批次
x = ones(d,1)*(ksel/d);


x_track = []; 
x_ave_track = [];  
eps_track = []; 
obj_track = func_global_SenSel(A_allT, x);



old_grad = 0;


for t = 1 :IterMax    %% 总迭代次数
    
    grad = 0;
    if mod(t-1,q) == 0   %% 收计算满梯度
        idx_sample = randperm(N,B1); % N=1000， 从1-1000里随机选取 Bt个数
        A_sel = A_allT(:,:,idx_sample);   % 将选中的数据提取出来
        for i = 1:B1    %% 便利所有的函数,接下来对单个函数怎么求出梯度
            E = eye(d);  % 50*50单位矩阵
            for j =1:d
                deta = E(:,j);
                [f_temp1 , grad_temp1] = func_loc_n_SenSel(A_sel(:,:,i),x+mu*deta);
                [f_temp2 , grad_temp2] = func_loc_n_SenSel(A_sel(:,:,i),x-mu*deta);
                grad = grad + (f_temp1 - f_temp2)/(2*mu)*deta*(1/B1);
            end
        end
    else
        idx_sample = randperm(N,B2); % N=1000， 从1-1000里随机选取 Bt个数
        A_sel = A_allT(:,:,idx_sample);   % 将选中的数据提取出来
        grad1 = 0;
        grad2 = 0;
        for i = 1:B2
            E = eye(d);  % 50*50单位矩阵
            for j=1:d
                deta = E(:,j);
                [f_temp1 , grad_temp1] = func_loc_n_SenSel(A_sel(:,:,i),x+mu*deta);
                [f_temp2 , grad_temp2] = func_loc_n_SenSel(A_sel(:,:,i),x-mu*deta);
                grad1 = grad1 + (f_temp1 - f_temp2)/(2*mu)*deta*(1/B2);
                [f_temp1_old , grad_temp1] = func_loc_n_SenSel(A_sel(:,:,i),old_x+mu*deta);
                [f_temp2_old , grad_temp2] = func_loc_n_SenSel(A_sel(:,:,i),old_x-mu*deta);
                grad2 = grad2 + (f_temp1_old - f_temp2_old)/(2*mu)*deta*(1/B2);
            end
        end
        grad = grad1-grad2+old_grad;
    end
    old_grad = grad;  % 保存旧梯度
    old_x = x ;       % 保存旧参数
    w = LMO(grad,ksel,d);
    v = w - x;
    x = x + lr*v;
    
    %%% 保存信息
    x_track = [x_track,x];
    x_ave = mean(x_track,2);
    x_ave_track = [x_ave_track,x_ave];
    
    if t > 1
            eps_track = [eps_track; norm(x_ave_track(:,end)-x_ave_track(:,end-1))]; %% 这一行计算最近两次迭代中 x_ave_track 的变化量，并添加到 eps_track 中。
            obj_temp = func_global_SenSel(A_allT, x_ave);  %% 计算目标函数值
            obj_track = [obj_track;  obj_temp];           %% 追踪 目标函数值
            if mod(t,50) == 0    %% 每50打印一次信息
               disp(sprintf('FZFW for iter = %d with xeps = %4.5f, obj = %4.5f',...
                    t, eps_track(end),obj_track(end))); 
            end
    end
end


end