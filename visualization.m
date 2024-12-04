clc;
clear all;
load("uci-digit.mat")
%  X=fea;
% Y=gt;
 k=length(unique(Y));
 num_view = length(X);
 for v=1:num_view
     X{v}=X{v}';
 end
num_sample = size(X{1},2);
view_Group=[];
 anchor=3*k;
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
        view_distance(i,j) = norm(WS{i}-WS{j},'fro')^2;
    end
end
view_distance=(view_distance+view_distance')/2;
sigma = mean(mean(view_distance));
dist_matrix=view_distance/sigma;
sm_matrix=ones(num_view,num_view)./(ones(num_view,num_view)+dist_matrix);
mean_sm_matrix=sum(sm_matrix)./num_view;

%% Quickshift++
% compute r_K
K=2;
alpha=[0.1 0.3 0.5 0.7 0.9];
for al=1:length(alpha)
% r_K=zeros(num_view,1);
% threshold=3;
% for i=1:num_view
%     sort_dis=sort(dist_matrix(i,:));
%     sort_dis=sort_dis(2:K+1);
%     r_K(i)=max(sort_dis);
% end
% K_NN=zeros(num_view,num_view);
% sort_r_K=sort(r_K);
% for i=1:num_view
%     r=sort_r_K(i);
%     if r>threshold
%         break;
%     end
%     for t=1:num_view
%         if r_K(t)<=r
%             for j=1:num_view
%                 if dist_matrix(t,j)<=alpha*min(r_K(t),r_K(j))
%                     K_NN(t,j)=dist_matrix(t,j);
%                 end
%             end
%         end
%     end
% end
% K_NN_Graph=graph(K_NN);
% L = conncomp(K_NN_Graph);
res{al}=quickshift(dist_matrix,K,alpha(al));
end
%% 
X = cmdscale(dist_matrix,2);
%scatter(X(:, 1), X(:, 2));
colors = [142 207 201;
          255 190 122;
          250 127 111;
          %250 127 111;
          %130 176 210;
          %130 176 210;
          %130 176 210;
          ]/255; 

scatter(X(:, 1), X(:, 2), 100, colors, 'Marker', 'o', 'MarkerEdgeColor', 'white', 'MarkerFaceColor', 'flat');

for i = 1:size(X, 1)
    text(X(i, 1), X(i, 2), num2str(i), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 5, 'Color', 'black');
end

xlabel('uci-digit', 'FontSize', 50);
axis equal;

% 
xRange = [min(X(:, 1)), max(X(:, 1))];
yRange = [min(X(:, 2)), max(X(:, 2))];
xMargin = 2 * diff(xRange);
yMargin = 2 * diff(yRange);
axis([xRange(1)-xMargin, xRange(2)+xMargin, yRange(1)-yMargin, yRange(2)+yMargin]);
box on;
% view_graph = exp(-view_distance/sigma);
% 
% idx = spectralcluster(view_graph,4,'Distance','precomputed')



