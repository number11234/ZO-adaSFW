function [x_ave_track, eps_track, obj_track,obj_track_50] = ACC_UNS_SenSel(options)
%ZO-ADASFW �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
A_allT = options.data;
d = size(A_allT,2); %%% number of optimization variables  50
N = size(A_allT,3); %%% number of random samples   1000
% x = options.x0;  % ��Ϊ����������
IterMax = options.IterMax ;   % ��������  
%eps = options.eps ;
ksel = options.ksel;   % ������������

mu = 1/(d*IterMax^(1/2));      % ƽ������
B1 = 50;
B2 = 50 ;      % ����
Q =  10;   % �ݶȹ��ƴ���   
eta = 1/(IterMax) ;    % ѧϰ��
    % frank ����
x = ones(d,1)*(ksel/d);
a = x ;
b = x ;



x_track = []; 
x_ave_track = [];  
eps_track = []; 
obj_track = func_global_SenSel(A_allT, x);
old_grad = 0;
obj_track_50 = func_global_SenSel(A_allT, x)/1000;


for t = 1 :IterMax    %% �ܵ�������
    grad = 0;
    ztj_sub_batch = [];
    for j = 1:Q    % Q�ݶȹ��Ƴ�������
         ztj = rand(size(x))*2-1;
         norm_value = norm(ztj, 2);
         ztj = ztj/norm_value;
         ztj_sub_batch = [ztj_sub_batch,ztj];
    end
    rho = 1/(t)^(2/3);
    if t == 1   %% �ռ������ݶ�
        idx_sample = randperm(N,B1); % N=1000�� ��1-1000�����ѡȡ Bt����
        A_sel = A_allT(:,:,idx_sample);   % ��ѡ�е�������ȡ����
       
        for i = 1:B1    %% �������еĺ���,�������Ե���������ô����ݶ�
            for j = 1:Q
                ztj = ztj_sub_batch(:,j);
                [f_temp1 , grad_temp1] = func_loc_n_SenSel(A_sel(:,:,i),x+mu*ztj);
                [f_temp2 , grad_temp2] = func_loc_n_SenSel(A_sel(:,:,i),x);
                grad = grad + (f_temp1 - f_temp2)/(mu)*ztj*d*(1/(B1*Q));
            end
        end
    else
        idx_sample = randperm(N,B2); % N=1000�� ��1-1000�����ѡȡ Bt����
        A_sel = A_allT(:,:,idx_sample);   % ��ѡ�е�������ȡ����
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
    old_x = x ; % ����ɲ���
    old_grad = grad;  % ������ݶ�
    w = LMO(grad, ksel,d);
    a = a + gamma * (w - a);
    b = x + eta*(w-x);
    x = (1-alpha)*b + alpha*a ;
  
   
    %%% ������Ϣ
    x_track = [x_track,x];
    x_ave = mean(x_track,2);
    x_ave_track = [x_ave_track,x_ave];
    
    if t > 1
            eps_track = [eps_track; norm(x_ave_track(:,end)-x_ave_track(:,end-1))]; %% ��һ�м���������ε����� x_ave_track �ı仯��������ӵ� eps_track �С�
            obj_temp = func_global_SenSel(A_allT, x_ave);  %% ����Ŀ�꺯��ֵ
            obj_track = [obj_track;  obj_temp];           %% ׷�� Ŀ�꺯��ֵ
            if mod(t,50) == 0    %% ÿ50��ӡһ����Ϣ
               disp(sprintf('ACC_UNS for iter = %d with xeps = %4.5f, obj = %4.5f',...
                    t, eps_track(end),obj_track(end))); 
                 obj_track_50 = [obj_track_50;  obj_temp/1000];
            end
    end
end


end