function [x_ave_track, eps_track, obj_track,obj_track_50] = ZSCG_SenSel(options)
    %ZO-ADASFW �˴���ʾ�йش˺�����ժҪ
    %   �˴���ʾ��ϸ˵��
    A_allT = options.data;
    d = size(A_allT,2); %%% number of optimization variables  50
    N = size(A_allT,3); %%% number of random samples   1000
    % x = options.x0;  % ��Ϊ����������
    IterMax = options.IterMax ;   % ��������  
    %eps = options.eps ;
    ksel = options.ksel;
    Q =  10;   % �ݶȹ��ƴ���
    Bt =  50 ;  % ��������
    x = ones(d,1)*(ksel/d);
    mu = 1/(sqrt(d^3)*IterMax);
    lr = 1/IterMax^(3/4);
    x_track = []; 
    x_ave_track = [];  
    eps_track = []; 
    obj_track = func_global_SenSel(A_allT, x);
    obj_track_50 = func_global_SenSel(A_allT, x)/1000;
    for t = 1 :IterMax    %% �ܵ�������
        
        % ut = grad_est_const/(t*d);
        idx_sample = randperm(N,Bt);  % ���ݳ���
        A_sel = A_allT(:,:,idx_sample); 
        ztj_sub_batch = [];
        grad = 0 ;

        for j = 1:Q    % Q�ݶȹ��Ƴ�������
            ztj = mvnrnd(zeros(1,d),eye(d)).';
            ztj_sub_batch = [ztj_sub_batch,ztj];
        end
        for i = 1:Bt
            for j = 1:Q
                ztj = ztj_sub_batch(:,j);
                [f_temp1 , grad_temp1]= func_loc_n_SenSel(A_sel(:,:,i), x+mu*ztj);
                [f_temp2 , grad_temp2]= func_loc_n_SenSel(A_sel(:,:,i),  x);
                grad = grad + (f_temp1 - f_temp2)/mu*ztj*(1/(Bt*Q));
            end
        end

        w = LMO(grad,ksel,d);

        v = w - x;
        x = x + lr*v ;   
        %%% ������Ϣ
        x_track = [x_track,x];
        x_ave = mean(x_track,2);
        x_ave_track = [x_ave_track,x_ave];

        if t > 1
                eps_track = [eps_track; norm(x_ave_track(:,end)-x_ave_track(:,end-1))]; %% ��һ�м���������ε����� x_ave_track �ı仯��������ӵ� eps_track �С�
                obj_temp = func_global_SenSel(A_allT, x_ave);  %% ����Ŀ�꺯��ֵ
                obj_track = [obj_track;  obj_temp];           %% ׷�� Ŀ�꺯��ֵ
                if mod(t,50) == 0    %% ÿ50��ӡһ����Ϣ
                   disp(sprintf('ZSCG for iter = %d with xeps = %4.5f, obj = %4.5f',...
                        t, eps_track(end),obj_track(end)));
                    obj_track_50 = [obj_track_50;  obj_temp/1000]; 
                end
        end
    end
     
            
        
    
    
    

