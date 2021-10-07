function [regret, pulls] = LinUCB(arms,std,K,T,d,xi)
%     The function implements LinUCB policy for contextual multi-armd 
%     bandit problem. 
%     Input :
%         arms: the mean reward for each arm
%         std: standard deviation
%         K: number of arms
%         T: number steps
%         xi: parameter for LinUCB policy
%         d: the dimension of the contextual feature vector  
%     Output: 
%         regret: regret for each round. row vector 
%         pulls: sequential arms chosen by LinUCB algorithm at each time

% initialization
load training_data  %include the contextual feature vector data at each time for each arm
ucb = zeros(K,1);
alpha=1+sqrt(0.5*(log(2/xi)));
pulls = zeros(1,T);
regret=zeros(1,T);
optimal=max(arms);
for i=1:K
     A(:,:,i) = eye(d);
     b(:,:,i)=zeros(d,1);
     theta(:,:,i)=zeros(d,1);
end
for t=1:T
    for i=1:K
         Ainv=inv(A(:,:,i));
         theta(:,:,i)=Ainv*b(:,:,i);
         ucb(i)=theta(:,:,i)'*x(:,t,i)+alpha*sqrt(x(:,t,i)'*Ainv*x(:,t,i));
    end
    [idx,~]  = max(ucb);
    l= x(:,t,idx); 
    reward = normrnd(arms(idx),std);%
    % tracking
    pulls(1,t) = idx;
    regret(1,t)=optimal-arm(idx);
    %online updatees
    A(:,:,idx)=A(:,:,idx)+x(:,t,idx)*x(:,t ,idx)';
    b(:,:,idx)=b(:,:,idx)+reward*x(:,t,idx);
end

for t=1:T
    Regret(t)=sum(regret(1:t))/log(t);
end
    
