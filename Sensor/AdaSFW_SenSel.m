function [x_ave_track, eps_track, obj_track,obj_track_50] = AdaSFW_SenSel(options)
%ZO-ADASFW �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
A_allT = options.data;
d = size(A_allT,2); %%% number of optimization variables  50
N = size(A_allT,3); %%% number of random samples   1000
% x = options.x0;  % ��Ϊ����������
IterMax = options.IterMax ;   % ��������  
%eps = options.eps ;
ksel = options.ksel;   % ������������
B1 = 50;
ada_eps = options.ada_eps;  % ��Сֵ��ֹ����Ϊ0
K = options.K ;        % frank ��������
eta = 1 ;    % ѧϰ��
base_gamma = options.base_gamma ;    % frank ����
x = ones(d,1)*(ksel/d);
obj_track_50 = func_global_SenSel(A_allT, x)/1000;

x_track = []; 
x_ave_track = [];  
eps_track = []; 
obj_track = func_global_SenSel(A_allT, x);
old_grad = 0;
accumulator = 0;

for t = 1 :IterMax    %% �ܵ�������
    grad = 0;
    if mod(t,100)==0
        eta = eta/2;
        if eta <= 0.1
            eta =0.1;
        end
    end
    idx_sample = randperm(N,B1); % N=1000�� ��1-1000�����ѡȡ Bt����
    A_sel = A_allT(:,:,idx_sample);   % ��ѡ�е�������ȡ����
    for i = 1:B1    %% �������еĺ���,�������Ե���������ô����ݶ�
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
    x = y;     % �����±���
    %%% ������Ϣ
    x_track = [x_track,x];
    x_ave = mean(x_track,2);
    x_ave_track = [x_ave_track,x_ave];
    
    if t > 1
            eps_track = [eps_track; norm(x_ave_track(:,end)-x_ave_track(:,end-1))]; %% ��һ�м���������ε����� x_ave_track �ı仯��������ӵ� eps_track �С�
            obj_temp = func_global_SenSel(A_allT, x_ave);  %% ����Ŀ�꺯��ֵ
            obj_track = [obj_track;  obj_temp];           %% ׷�� Ŀ�꺯��ֵ
            if mod(t,50) == 0    %% ÿ50��ӡһ����Ϣ
               disp(sprintf('adaSFW for iter = %d with xeps = %4.5f, obj = %4.5f',...
                    t, eps_track(end),obj_track(end))); 
                obj_track_50 = [obj_track_50;  obj_temp/1000];
            end
    end
end


end