function [regret, pulls]=Exp3(K,arms,std, T)
%     The function implements Exp3 policy for multi-armd 
%     bandit problem. 
%     Input :
%         arms: the mean reward for each arm
%         K: number of arms
%         std: standard deviation
%         T: number steps
%     Output: 
%         regret: regret for each round. row vector 
%         pulls: sequential arms chosen by Exp3

optimal = max(arms);         % the reward for optimal arm
pulls = zeros(1,T);          % initialize pulls
regret = zeros(1,T);
gamma=0.01;
% initialize w. Set as 1 for each w(i)
w = zeros(1,K);
for i=1:K
    w(1,i) = 1;
end
pro = zeros(1,K);
% iterate
for iter=1:T
    % Set probability    
    for i=1:K
        pro(1,i)=(1-gamma)*w(1,i)/sum(w(1,:)) + gamma/K;
    end
    % decide which arm to pull based on weights
    idx = getIndexFromProbability(pro,K);
    % pull the arms with index idx
    regret(1,iter) = optimal-arms(idx);
    pulls(1,iter) = idx;  
    reward = normrnd(arms(idx),std);
    % update the probabilit for each arm
    for i=1:K
        if i == idx
            w(i,1)=w(i,1)*exp((gamma*reward/pro(i,1))/K);
        end
    end
end
for t=1:T
    Regret(t)=sum(regret(1:t))/log(t);
end
