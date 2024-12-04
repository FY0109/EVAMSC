function [L,output]=VCMSC(X,s,d,l,k,lambda,options)
% 1/2*alpha^2*|X(v)_t-Hg*Pg|+beta*|Pg-CgAS|+lambda*|S-G*F|
% X: m*n
% s: number of community
% d: number of anchor
% l: number of bases
% k: cluster number
% lambda: parameters
% options: maxiter,isnorm

anchor=3*k;
if isfield(options,'maxiter') maxiter=options.maxiter;else maxiter=100;end
%if isfield(options,'isreverse') maxiter=options.isreverse;else maxiter=1;end
%% initialize 
num_view = length(X);
num_sample = size(X{1},2);

%% compute view self-representation
% find anchor
X_total=X{1};
feature=[size(X{1},1)];
for v=2:num_view
    X_total=cat(1, X_total, X{v});
    feature(v)=size(X{v},1);
end
 kmMaxIter = 5;
 kmNumRep = 1;
 [~,marks]=litekmeans(X_total',anchor,'MaxIter',kmMaxIter,'Replicates',kmNumRep);
 start=1;
for v=1:num_view
    mark_Z=marks(:,start:start+feature(v)-1);
    start=feature(v)+1;
    D = EuDist2(X{v}',mark_Z,0);
    sigma = mean(mean(D));
    D = exp(-D/abs(sigma));
    sumD=sum(D);
    temp=repmat(sumD,num_sample,1);
    WS{v}=D./temp;
    
end
view_distance=zeros(num_view,num_view);
for i=1:num_view
    for j=1:num_view
        view_distance(i,j) =  norm(WS{i}-WS{j},'fro')^2;
    end
end
 for v=1:num_view
        X{v} = zscore(X{v});
 end

view_distance=(view_distance+view_distance')/2;
sigma = mean(mean(view_distance));
dist_matrix=view_distance/sigma;
%idx = spectralcluster(view_graph,s,'Distance','precomputed');
idx=quickshift(dist_matrix,num_view-1,0.3);

alpha = zeros(s,num_view);

beta = ones(s,1)*sqrt(1/s);
Z = cell(s,1);
H = cell(s,num_view);
m=k;
cnt = zeros(s,num_view);
cnt1 = ones(s,1);
P=cell(s,1);
for i=1:s
    Z{i} = eye(3*m,num_sample);
    P{i} = eye(3*m,d);
    for j=1:num_view
        if idx(j)==i
            sub_view=sum(idx == idx(j));
            alpha(i,j)=1/sub_view;
        end
    end
end

A=eye(d,l);
S=zeros(l,num_sample);
S(:,1:l) = eye(l);
G = eye(l,k);
F = eye(k,num_sample); 

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
%% Update H
    
    H = update_H(X,Z,H,s,idx);
    
%% Update Z
    
    Z = update_Z(X,Z,H,P,A,S,alpha,beta,s,idx);
    

%% Update P_i
    
    AS = A*S; 
    parfor iv=1:s
        C = Z{iv}*AS';      
        [U,~,V] = svd(C,'econ');
        P{iv} = U*V';
    end
   
%% Update A
    
    sumAlpha = 0;
    part1 = 0;
    for ia = 1:s
        al2 = alpha(ia)^2;
        sumAlpha = sumAlpha + al2;
        part1 = part1 + al2 * P{ia}' * Z{ia} * S';
    end
    [Unew,~,Vnew] = svd(part1,'econ');
    A = Unew*Vnew';
    
%% Update S
    
    HS = 2*sumAlpha*eye(l)+2*lambda*eye(l);
    HS = (HS+HS')/2;
    options = optimset( 'Algorithm','interior-point-convex','Display','off'); % interior-point-convex
    parfor ji=1:num_sample
        ff=0;
        e = F(:,ji)'*G';
        for j=1:s
            C = P{j} * A;
            ff = ff - 2*Z{j}(:,ji)'*C - 2*lambda*e;
        end
        S(:,ji) = quadprog(HS,ff',[],[],ones(1,l),1,zeros(l,1),ones(l,1),[],options);
    end
   

    
%% Update G
    S_normlize=zscore(S);
    %S_normlize=S;
    J = S_normlize*F';      
    [Ug,~,Vg] = svd(J,'econ');
    G = Ug*Vg';
    
    %% Update F
   
    F=zeros(k,num_sample);
    for iff=1:num_sample
        Dis=zeros(k,1);
        for jf=1:k
            Dis(jf)=(norm(S(:,iff)-G(:,jf)))^2;
        end
        [~,r]=min(Dis);
        F(r(1),iff)=1;
    end
 
%% Update alpha beta
        
    for i=1:s
        cnt1(i)=norm(Z{i}-P{i}*A*S,'fro')^2;
        for j=1:num_view
            if idx(j)==i
            cnt(i,j) = norm(X{j}-H{i,j}*Z{i},'fro')^2;
            end
        end
    end

    alpha = update_alpha(cnt,s,num_view);
    beta = update_beta(cnt1,s);

%% compute obj
    
    
   
    obj(iter) = cal_obj(cnt,cnt1,alpha,beta);
    obj(iter)=obj(iter)+lambda*norm(S - G * F,'fro')^2;
    if (iter>2) && (abs((obj(iter)-obj(iter-1))/(obj(iter)))<1e-5 || iter>maxiter)
        flag =0;
    end
    
end

%% classifier

[~,L]=max(F);
output.S=S;
output.F=F;
output.loss=obj;

end

%%


%% 
function [H] = update_H(fea,Z,H,s,idx)

num_view = length(fea);
T = cell(s,num_view);
for i=1:s
    for j=1:num_view
        if idx(j)==i
            T{i,j} = fea{j};
            H{i,j} = T{i,j}*Z{i}';
        end
    end
end
end


%%
function [alpha] = update_alpha(cnt,s,num_view)

tmp=cnt;
tmp = ones(s,num_view)./tmp;
tmp(isinf(tmp)) = 0;
total = sum(tmp,2);
for i=1:s
    alpha(i,:) = tmp(i,:)/total(i);
end
end

%%
function [beta] = update_beta(cnt1,num_p)

cnt = 0;
for i=1:num_p
    cnt = cnt+cnt1(i)^2;
end
beta = cnt1/sqrt(cnt);
end
%%
function [Z] = update_Z(fea,Z,H,P,A,S,alpha,beta,s,idx)

num_view = length(fea);
U = cell(s,1);
V = cell(s,1);
T = cell(s,num_view);
M = cell(s,1);
CNT = cell(s,1);
for i=1:s
    CNT{i} = zeros(size(Z{1},2),size(Z{i},1));
    for j=1:num_view
        if idx(j)==i
            T{i,j} = fea{j};
            CNT{i} = CNT{i}+alpha(i,j)^2*T{i,j}'*H{i,j};
        end
    end
    CNT{i} = CNT{i}+beta(i)*S'*A'*P{i}';
    [U{i},~,V{i}] = svd(CNT{i},'econ');
    M{i} = U{i}*V{i}';
    Z{i} = M{i}';
end
end

%%
function [obj] = cal_obj(cnt,cnt1,alpha,beta)

obj = sum((1/2*alpha.^2).*cnt,'all')+sum(beta.*cnt1);

end


