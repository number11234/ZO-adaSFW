function [x_ave_track, eps_track, obj_track,obj_track_50] = AdaSFW_SenSel(options)
%ZO-ADASFW 此处显示有关此函数的摘要
%   此处显示详细说明
A_allT = options.data;
d = size(A_allT,2); %%% number of optimization variables  50
N = size(A_allT,3); %%% number of random samples   1000
% x = options.x0;  % 改为符合条件的
IterMax = options.IterMax ;   % 迭代数量  
%eps = options.eps ;
ksel = options.ksel;   % 传感器的数量
B1 = 50;
ada_eps = options.ada_eps;  % 极小值防止开方为0
K = options.K ;        % frank 迭代次数
eta = 1 ;    % 学习率
base_gamma = options.base_gamma ;    % frank 参数
x = ones(d,1)*(ksel/d);
obj_track_50 = func_global_SenSel(A_allT, x)/1000;

x_track = []; 
x_ave_track = [];  
eps_track = []; 
obj_track = func_global_SenSel(A_allT, x);
old_grad = 0;
accumulator = 0;

for t = 1 :IterMax    %% 总迭代次数
    grad = 0;
    if mod(t,100)==0
        eta = eta/2;
        if eta <= 0.1
            eta =0.1;
        end
    end
    idx_sample = randperm(N,B1); % N=1000， 从1-1000里随机选取 Bt个数
    A_sel = A_allT(:,:,idx_sample);   % 将选中的数据提取出来
    for i = 1:B1    %% 便利所有的函数,接下来对单个函数怎么求出梯度
        [f_temp , grad_temp] = func_loc_n_SenSel(A_sel(:,:,i),x);
        grad = grad + grad_temp*(1/B1);
    end
    accumulator = accumulator + grad.*grad;;
    H2 = ada_eps + sqrt( accumulator); 
    %H1 = clip(H2,0,30);
    H = diag(H2);
    y = x;
    gamma_t = base_gamma/(t+1);
    for k=1:K
        nabla_Z = grad + H*(y-x)/eta;
        w = LMO(nabla_Z, ksel,d);
        gamma_k = min( dot(nabla_Z,(y-w))*eta/((y-w).'*H*(y-w)), gamma_t  );
        y = y + gamma_k*(w-y);
    end
    x = y;     % 更新新变量
    %%% 保存信息
    x_track = [x_track,x];
    x_ave = mean(x_track,2);
    x_ave_track = [x_ave_track,x_ave];
    
    if t > 1
            eps_track = [eps_track; norm(x_ave_track(:,end)-x_ave_track(:,end-1))]; %% 这一行计算最近两次迭代中 x_ave_track 的变化量，并添加到 eps_track 中。
            obj_temp = func_global_SenSel(A_allT, x_ave);  %% 计算目标函数值
            obj_track = [obj_track;  obj_temp];           %% 追踪 目标函数值
            if mod(t,50) == 0    %% 每50打印一次信息
               disp(sprintf('adaSFW for iter = %d with xeps = %4.5f, obj = %4.5f',...
                    t, eps_track(end),obj_track(end))); 
                obj_track_50 = [obj_track_50;  obj_temp/1000];
            end
    end
end


end