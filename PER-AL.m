function [regret, pulls]=PER-AL(K,arms,std,Rounds,T_m,T_e)
%     The function implements PER-AL policy for adversarial multi-armd 
%     bandit-model problem. 
%     Input :
%         K: number of arms
%         arms: the mean reward for each arm
%         std: standard deviation
%         Rounds: number rounds in expert assisted learning
%         T_m: number of time steps in each round in expert assisted learning
%         T_e: number of time steps for self-adjusting learning
%     Output: 
%         regret: regret for each time. row vector
%         pulls: sequential arms chosen by PER-AL algorithm for each time

% Iinitialization
 optimal=max(arms);
 N=K;
 n=3;
 d_f=Rounds*T_m/10*K;
 gamma=0.01;
 mean_parm=ones(1,K);
 std_parm=zeros(1,K);
 T=Rounds*T_m+T_e;
 pro = zeros(1,K);
 regret = zeros(1,T); 
 pulls=zeros(1,T); 
 est_x=zeros(K,1);
Exp=rand(N,K); 
for j=1:N
    M=sum(Exp(j,:));
    for i=1:K
        Exp(j,i)=Exp(j,i)/M;
    end
end

%  iterate 
for r=1:Rounds        
w_expert=ones(1,N);
  for t=1:T_m
     W=sum(w_expert); 
     for i=1:K
     pro(1,i)=(1-gamma)*(w_expert*Exp(:,i)/W)+gamma/K;
     end
     idx = getIndexFromProbability(pro,K); 
     t_idx=T_m*(r-1)+t; 
     pulls(1,t)=idx;
     regret(1,t_idx)=optimal-arms(idx);
     reward = normrnd(arms(idx,1),std); 
     for i=1:K
         if i==idx
             est_x(idx,1)=reward/pro(1,idx);
         else
             est_x(i,1)=0;
         end
     end
    for j=1:N
     Y(j)=Exp(j,:)*est_x;     
     w_expert(j)=w_expert(j)*exp((gamma*Y(j))/K);
    end
end
% learn the optimal expert and define the probability distribution
 [~,I]=max(w_expert);
 opti_exp=Exp(I,:);
 up=ones(1,K);
 dw=zeros(1,K);
   for i=1:K
    mean_parm(1,i)= opti_exp(1,i);
    std_parm(1,i) =(min(mean_parm(1,i)-dw(1,i),up(1,i)-mean_parm(1,i)))/2;
   end
   for i=1:K
   adv_parm(i,:)=normrnd(mean_parm(i),std_parm(i),n);  %sample n value for each advice parameter     
   end
% reconstruction new expert with respect to K
%i.e.,K=4
[q4 q3 q2 q1] = ndgrid(adv_parm(4,:),adv_parm(3,:),parm_adv(2,:),parm_adv(1,:)); 
new_exp=[q1(:) q2(:) q3(:) q4(:)];
new_exp = unique(new_exp,'rows'); 
Exp=[Exp;new_exp];
[Nnew_exp,~]=size(newExp);
N=N+Nnew_exp;
end

w_arms=d_f*opti_exp; 
for t=1:T_e
    W=sum(w_expert);
    for i=1:K
        pro(1,i)=(1-gamma)*w_arms(i)/W+(gamma/K);
    end
    idx = getIndexFromProbability(pro,k);  
    t_idx=Rounds*T_m+t;
    regret(1,t_idx)=optimal-arms(idx); 
    pulls(1,t_idx)=idx;
    reward=normrnd(arms(idx),std);
    w_arms(idx)=w_arms(idx)*exp((gamma*reward)/K/pro(1,idx));
end

for t=1:T
    Regret(t)=sum(regret(1:t))/log(t);
end





