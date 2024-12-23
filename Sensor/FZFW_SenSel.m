function [x_ave_track, eps_track, obj_track] = FZFW_SenSel(options)
%ZO-ADASFW �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
A_allT = options.data;
d = size(A_allT,2); %%% number of optimization variables  50
N = size(A_allT,3); %%% number of random samples   1000
% x = options.x0;  % ��Ϊ����������
IterMax = options.IterMax ;   % ��������  
%eps = options.eps ;
ksel = options.ksel;   % ������������
q = 2;         % �����ݶ�
mu = 1/(sqrt(IterMax*d)) ;      % ƽ������
lr = 1/(d*sqrt(IterMax));
B1 = 50;
B2 = 50 ;      % ����
x = ones(d,1)*(ksel/d);


x_track = []; 
x_ave_track = [];  
eps_track = []; 
obj_track = func_global_SenSel(A_allT, x);



old_grad = 0;


for t = 1 :IterMax    %% �ܵ�������
    
    grad = 0;
    if mod(t-1,q) == 0   %% �ռ������ݶ�
        idx_sample = randperm(N,B1); % N=1000�� ��1-1000�����ѡȡ Bt����
        A_sel = A_allT(:,:,idx_sample);   % ��ѡ�е�������ȡ����
        for i = 1:B1    %% �������еĺ���,�������Ե���������ô����ݶ�
            E = eye(d);  % 50*50��λ����
            for j =1:d
                deta = E(:,j);
                [f_temp1 , grad_temp1] = func_loc_n_SenSel(A_sel(:,:,i),x+mu*deta);
                [f_temp2 , grad_temp2] = func_loc_n_SenSel(A_sel(:,:,i),x-mu*deta);
                grad = grad + (f_temp1 - f_temp2)/(2*mu)*deta*(1/B1);
            end
        end
    else
        idx_sample = randperm(N,B2); % N=1000�� ��1-1000�����ѡȡ Bt����
        A_sel = A_allT(:,:,idx_sample);   % ��ѡ�е�������ȡ����
        grad1 = 0;
        grad2 = 0;
        for i = 1:B2
            E = eye(d);  % 50*50��λ����
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
    old_grad = grad;  % ������ݶ�
    old_x = x ;       % ����ɲ���
    w = LMO(grad,ksel,d);
    v = w - x;
    x = x + lr*v;
    
    %%% ������Ϣ
    x_track = [x_track,x];
    x_ave = mean(x_track,2);
    x_ave_track = [x_ave_track,x_ave];
    
    if t > 1
            eps_track = [eps_track; norm(x_ave_track(:,end)-x_ave_track(:,end-1))]; %% ��һ�м���������ε����� x_ave_track �ı仯��������ӵ� eps_track �С�
            obj_temp = func_global_SenSel(A_allT, x_ave);  %% ����Ŀ�꺯��ֵ
            obj_track = [obj_track;  obj_temp];           %% ׷�� Ŀ�꺯��ֵ
            if mod(t,50) == 0    %% ÿ50��ӡһ����Ϣ
               disp(sprintf('FZFW for iter = %d with xeps = %4.5f, obj = %4.5f',...
                    t, eps_track(end),obj_track(end))); 
            end
    end
end


end