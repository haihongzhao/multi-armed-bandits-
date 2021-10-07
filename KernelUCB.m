function [regret, pulls]=KernelUCB(arms,std,K,T, lambda,delta)
%     The function implements KernelUCB policy for multi-armd 
%     bandit-model problem. 
%     Input :
%         K: number of arms
%         arms: the mean reward for each arm
%         std: standard deviation
%         T: time steps 
%         lambda: parameter in KernelUCB policy
%         delta:  parameter in KernelUCB policy
%     Output: 
%         regret: regret for each time. row vector
%         pulls: sequential arms chosen by KernelUCB algorithm for each time
% Iinitialization
load training_data  %include the contextual feature vector data at each time for each arm
 ucb=zeros(K,1);
 optimal=max(arms);
 predictions = zeros(K,1);
 variances = zeros(K,1);
 regret=zeros(T,1);
 pulls = zeros(T,1);
 rewards = zeros(T,1);   
    for t=1:T
     Kgood=max(ucb);
    [Kh,~]=find(ucb==Kgood);
    it=Kh;
    rt = normrnd(arms(it),std);
    pulls(t,1) = it;
    rewards(t) = rt;
    regret(t)=optimal-arms(it);
    y=rewards(1:t,1);
    for i=1:K
    D(i,:)=x(i,:);
    end
    kx = DK(it,pulls(1:t))';
    % Compute Regularised Kernel Inverse    
    if length(kx)==1
        Kinv = 1./(kx + lambda);
    else
        b = kx(1:end-1);
        bKinv = b'*Kinv;
        Kinvb = Kinv*b;       
        K22 = 1./((kx(end) + lambda) - bKinv*b);
        K11 = Kinv + K22*Kinvb*bKinv;
        K12 = -K22*Kinvb;
        K21 = -K22*bKinv;
        Kinv = [K11 K12; K21 K22];
    end
    eta = sqrt((log(2*T*K/delta)/(2*lambda)));
    % compute UCBs  
    for i = 1:K
        kx = DK(i,pulls(1:t))';
        predictions(i) = kx'*Kinv*y;
        variances(i) = DK(i,i)-kx'*Kinv*kx;
        ucb(i) = predictions(i) + eta*sqrt(variances(i));
    end
    end
 for t=1:T
   Regret(t)=sum(regret(1:t))/log(t);
end
    