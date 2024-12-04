function L=quickshift(dist_matrix,K,alpha)
[num_view,~]=size(dist_matrix);
r_K=zeros(num_view,1);
threshold=3;
for i=1:num_view
    sort_dis=sort(dist_matrix(i,:));
    sort_dis=sort_dis(2:K+1);
    r_K(i)=max(sort_dis);
end
K_NN=zeros(num_view,num_view);
sort_r_K=sort(r_K);
for i=1:num_view
    r=sort_r_K(i);
    if r>threshold
        break;
    end
    for t=1:num_view
        if r_K(t)<=r
            for j=1:num_view
                if dist_matrix(t,j)<=alpha*min(r_K(t),r_K(j))
                    K_NN(t,j)=dist_matrix(t,j);
                end
            end
        end
    end
end
K_NN_Graph=graph(K_NN);
L = conncomp(K_NN_Graph);
end
