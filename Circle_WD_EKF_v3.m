clear all; close all;
%%
% @ Copyright Zhengyu Feng @ UESTC.
% @ Date 2023.04.24.
% @ Version V_1.0.
% @ 程序说明：本程序是以典型的非线性运动，从模型不确定性角度提升EKF算法的估计精度，推导一种 Wasserstein Distance (WD) EKF。
% @ 主要思路：用WD允许领域对EKF状态转移矩阵非线性的摄动求逆导致的估计精度下降的问题进行容忍，以提高EKF算法的估计精度。

%% 初始化参数
n = 2;  % 状态维数
m = 1;  % 观测维数
N = 1000;  % 时间

r = 50;  % 圆周运动半径
theta = 2*pi/36000;  % 每时刻转动的圆周角度

xx = ones(n,N);
xx(:,1) = [0.1, 0.1]';
% xx(:,1) = [1, 1]';
xe = ones(n,N);
% EKF 算法参数
xe_EKF = ones(n,N);
Pk_EKF = eye(n);

% WD EKF 算法参数
xe_MC2 = ones(n,N);
V_MCW2 = eye(n);
delta = 1;
Rho = 0.1;


KK = 50;
% T = 1;
Omega = 5 * 2* pi/360;
% Omega_T = Omega * T;
% F = [1, sin(Omega_T)/Omega, 0, -(1-cos(Omega_T))/Omega;
%     0, cos(Omega_T), 0, -sin(Omega_T);
%     0, (1-cos(Omega_T))/Omega, 1, sin(Omega_T)/Omega;
%     0, sin(Omega_T), 0, cos(Omega_T)];

H = [1, 0];
W = ones(1,N);
%% 算法迭代KK次
for kk = 1:KK
    % 状态高斯噪声
    q = randn(n,N)*0.01;
    Q = (q * q')/N;%% 必须知道状态噪声协方差 Q


    %% 节点1 参数初始化
    % v = randn(n,N)*1;
    q1 = randn(m,N)*1;q2 = randn(m,N)*10;
    r = randn(m,N) * 0.1;%%高斯噪声
%     vp = rand(m,N);
%     for jj = 1:m
%         for tt = 1:N
%             if vp(jj,tt) > 0.95
%                 r(jj,tt) = q2(jj,tt);
%             else
%                 r(jj,tt) = q1(jj,tt);
%             end
%         end
%     end

    R = (r * r')/N;
    %% 非线性运动
    for ii = 2:N
        %         F = [0.98, sin(xx(1,ii-1)); 0, 0.98];  %cos(ii*theta)cos(ii*theta)
        F = [0.8,  xx(1,ii-1);
            -xx(2,ii-1), 1.5];

        xx(:,ii) = F * xx(:,ii-1) + 0.1 + q(:,ii);% 状态方程
       
        yy(:,ii) = H * xx(:,ii) + r(:,ii);% 观测方程

        xee_EKF = [0.8 * xe_EKF(1,ii-1) +  xe_EKF(1,ii-1) * xe_EKF(2,ii-1);
            1.5 * xe_EKF(2,ii-1) - xe_EKF(1,ii-1) * xe_EKF(2,ii-1)]+ 0.1; 
        F = [0.8,  xee_EKF(1,:);
            -xee_EKF(2,:), 1.5];
%         cond(F)

%          sum(sum(abs(F-trace(F))))


        %         inv(F*F')
%         fprintf('EKF')
%         W(kk, ii) = sum(sum(inv(F)));


        %% EKF 算法
        Pke_EKF = F * Pk_EKF *F' + Q;
        K_EKF = Pke_EKF * H' * inv(H * Pke_EKF * H' + R);
        xe_EKF(:,ii) = F * xe_EKF(:,ii-1) + K_EKF * (yy(:,ii) - H * F * xe_EKF(:,ii-1));
        Pk_EKF = (eye(n) - K_EKF * H) * Pke_EKF * (eye(n) - K_EKF * H)' + K_EKF * R * K_EKF';
        K_EKF_C(:,ii) = sum(sum(K_EKF));%trace(Pk_EKF);
        % 估计误差
        Err_EKF(kk,ii) = norm(xe_EKF(:,ii) - xx(:,ii));

        %% WD EKF 算法
        yy_MC = yy(:,ii);
        xee_MC2 = xe_MC2(:,ii-1);
        xxee_WDEKF = [0.8*xe_MC2(1,ii-1) +  xe_MC2(1,ii-1) * xe_MC2(2,ii-1) + 0.1;
            1.5 * xe_MC2(2,ii-1) - xe_MC2(1,ii-1) * xe_MC2(2,ii-1) + 0.1];
        F = [0.8,  xxee_WDEKF(1,:);
            -xxee_WDEKF(2,:), 1.5];
%         cond(F)
%         fprintf('MU-EKF')
%         sum(sum(abs(F-trace(F))))
        % sigma = 2;
        
        [xee_MC2,V_MCW2,K_WD] = Function_WD_EKF_F(n,m,F,H,Q,R,Rho,delta,xee_MC2,yy_MC,V_MCW2);
        xe_MC2(:,ii) = xee_MC2;
        K_WDEKF_C(:,ii) = sum(sum(K_WD));%trace(V_MCW2);


        % 估计误差
        Err_MC_WKF2(kk,ii) = norm(xe_MC2(:,ii) - xx(:,ii));


    end
    fprintf('%d-th Iteration...\n',kk);
end

%% 画图
% figure; hold on;
% plot(xx(1,:),xx(2,:));%
% plot(yy(1,2:end),yy(2,2:end));
% plot(xe_EKF(1,:),xe_EKF(2,:));
% plot(xe_WD_EKF(1,:),xe_WD_EKF(2,:));
% legend('真实轨迹', '观测轨迹','EKF估计轨迹','WD-EKF估计轨迹');  %

figure; hold on;
plot(K_EKF_C(1,:));plot(K_WDEKF_C(1,:));
legend('Sum(P-EKF)','Sum(P-WD)');

figure; hold on;
plot(10*log10(mean(Err_EKF)));plot(10*log10(mean(Err_MC_WKF2)));
legend('EKF 估计误差','WD-EKF 估计误差');
% 
% figure; hold on;
% plot(mean(W));
% legend('F的逆');
% 
% 
[mean(mean(Err_EKF)),mean(mean(Err_MC_WKF2))]