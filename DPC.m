classdef DPC
    properties
        n_clusters
        dc
        percent
        dc_method
        rho_method
        delta_method
        threshold
        assign_method
        verbose
        rank
        dists
        rho_
        delta_
        ndh_
        centers_idx
        centers_
        labels_
    end

    methods
        function obj = DPC(n_clusters, dc, dc_method, rho_method, delta_method, dc_percent, assign_method, threshold, verbose, rank)
            if nargin < 10, rank = []; end
            if nargin < 9, verbose = false; end
            if nargin < 8, threshold = []; end
            if nargin < 7, assign_method = 0; end
            if nargin < 6, dc_percent = 1; end
            if nargin < 5, delta_method = 1; end
            if nargin < 4, rho_method = 2; end
            if nargin < 3, dc_method = 1; end
            if nargin < 2, dc = []; end
            if nargin < 1, n_clusters = []; end
            
            obj.n_clusters = n_clusters;
            obj.dc = dc;
            obj.percent = dc_percent;
            obj.dc_method = dc_method;
            obj.rho_method = rho_method;
            obj.delta_method = delta_method;
            obj.threshold = threshold;
            obj.assign_method = assign_method;
            obj.verbose = verbose;
            obj.rank = rank;
        end

        function obj = fit(obj, X, pre_density, dists)
            if nargin < 4
                dists = pdist2(X, X);
            end
            [n, ~] = size(X);
            
%             if isempty(obj.n_clusters) && isempty(obj.threshold)
%                 error('Either n_clusters or threshold must be set');
%             end
            
            obj.dists = dists;

            if obj.verbose, disp('GET RHO'); end
            if nargin >= 3 && ~isempty(pre_density)
                obj.rho_ = pre_density;
            else
                obj.rho_ = obj.get_rho(n);
            end

            if obj.verbose, disp('GET DELTA'); end
            [obj.delta_, obj.ndh_, obj.rank] = obj.get_delta(n);

            [obj.centers_idx, gamma] = obj.get_center();
            obj.centers_ = X(obj.centers_idx, :);

            if obj.verbose, disp('ASSIGN'); end
            obj.labels_ = obj.assign(n);
        end

        function dc = get_dc(obj, n)
            if obj.verbose, disp('get dc'); end
            dists = obj.dists;
            min_dis = min(dists(:));
            max_dis = max(dists(:));
            lower = obj.percent / 100;
            upper = (obj.percent + 1) / 100;

            flag = true;
            while flag
                dc = (min_dis + max_dis) / 2;
                neighbors_percent = (sum(sum(dists < dc)) - n) / ((n - 1)^2);
                if neighbors_percent >= lower && neighbors_percent <= upper
                    flag = false;
                elseif neighbors_percent > upper
                    max_dis = dc;
                else
                    min_dis = dc;
                end
            end
        end

        function rho = get_rho(obj, n)
            dists = obj.dists;
            rho = zeros(n, 1);

            switch obj.rho_method
                case 0
                    if isempty(obj.dc)
                        dc = obj.get_dc(n);
                    else
                        dc = obj.dc;
                    end
                    for i = 1:n
                        rho(i) = sum(dists(i,:) <= dc) - 1;
                    end

                case 1
                    if isempty(obj.dc)
                        dc = obj.get_dc(n);
                    else
                        dc = obj.dc;
                    end
                    for i = 1:n
                        rho(i) = sum(exp(-(dists(i,:) / dc).^2));
                    end

                case 2
                    sample_n = floor(n * 0.05);
                    for i = 1:n
                        sorted_d = sort(dists(i,:));
                        rho(i) = exp(-sum(sorted_d(1:sample_n)) / (sample_n - 1));
                    end

                case 3
                    knn_k = min(15, n);
                    for i = 1:n
                        sorted_d = sort(dists(i,:));
                        rho(i) = exp(-sorted_d(knn_k));
                    end

                case 4
                    sm_matrix = 1 ./ (1 + dists);
                    rho = sum(sm_matrix, 2);
            end
        end

        function [delta, ndh, rank] = get_delta(obj, n)
            rho = obj.rho_;
            dists = obj.dists;
            delta = zeros(n, 1);
            ndh = -ones(n, 1);
            rank = zeros(n, 1);

            if obj.delta_method == 0
                for i = 1:n
                    higher = find(rho >= rho(i));
                    if isempty(higher)
                        delta(i) = max(dists(:));
                    else
                        [val, idx] = min(dists(i, higher));
                        delta(i) = val;
                        ndh(i) = higher(idx);
                    end
                end

            elseif obj.delta_method == 1
                [~, rho_order] = sort(rho, 'descend');
                for i = 2:n
                    idx = rho_order(i);
                    higher = rho_order(1:i-1);
                    [val, j] = min(dists(idx, higher));
                    delta(idx) = val;
                    ndh(idx) = higher(j);
                    rank(idx) = higher(j);
                end
                delta(rho_order(1)) = max(delta);
                rank(rho_order(1)) = 1;

            else
                [~, rho_order] = sort(rho);
                for i = 1:n-1
                    idx = rho_order(i);
                    higher = rho_order(i+1:end);
                    [val, j] = min(dists(idx, higher));
                    delta(idx) = val;
                    ndh(idx) = higher(j);
                end
                delta(rho_order(end)) = max(delta);
            end
        end

        function [centers, gamma] = get_center(obj)
            gamma = obj.rho_ .* obj.delta_;
            [~, sorted_idx] = sort(gamma, 'descend');

            if ~isempty(obj.n_clusters)
                centers = sorted_idx(1:obj.n_clusters);
            else
                centers = find(gamma > obj.threshold);
            end
        end

        function labels = assign(obj, n)
            if obj.assign_method == 0
                labels = -1 * ones(n, 1);
                ndh = obj.ndh_;
                ndh(obj.centers_idx) = -1;

                for i = 1:length(obj.centers_idx)
                    center = obj.centers_idx(i);
                    labels(center) = i - 1;
                    idx = find(ndh == center);
                    while ~isempty(idx)
                        labels(idx) = i - 1;
                        idx = find(ismember(ndh, idx));
                    end
                end
            else
                d = obj.dists(:, obj.centers_idx);
                [~, labels] = min(d, [], 2);
                labels = labels - 1;
            end
        end
    end
end

