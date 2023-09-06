function [xee_W,V_W,K_WD] = Function_WD_EKF_F(n,m,F,H,Q,R,Rho,delta,xee_W,yy_W,V_W)
%MC_WKF_F �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
% prediction
% Form the pseudo-nominal distribution
% prediction
% Form the pseudo-nominal distribution
u = [F;H*F] * xee_W;% ���Ϸֲ�Ԥ���ֵ
Sigma = [F;H*F] * V_W * [F;H*F]' + [eye(n); H] * Q * [eye(n); H]';% * R���Ϸֲ�Ԥ��Э����
% ��Frank-Wolfe �㷨���
[S_star] = Frank_Wolfe_Algorithm(n,m,Sigma,u,Rho,delta);

%% ���
S_xx = S_star(1:n,1:n);
% S_xy = S_star(1:n,n+1:n+m);
% S_yx = S_star(n+1:n+m,1:n);
% S_yy = S_star(n+1:n+m,n+1:n+m);
u_x = u(1:n);
u_y = u(n+1:n+m);
% ����Э�������
K_WD = S_xx * H'*inv(H*S_xx*H' + R);%R LAMBDA_MC_yyBr*inv(LAMBDA_MC_yy)*Br'?????/

xee_W = K_WD * (yy_W  - u_y) + u_x;%B_r-  v(:,ii)

V_W =  (eye(n)-K_WD*H)*S_xx*(eye(n)-K_WD*H)' + K_WD*R*K_WD';


end

