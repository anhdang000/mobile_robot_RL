classdef Robot < Map
    % Mobile robot that works in the configured environment
    
    %   Define state properties
    %   'SS': a state where the robot has a low or no possibility
    %       of collision with some obstacles.
    %       (robot_pos is out of danger_range)
    %   'NS': a state where the robot has a high possibility of
    %       collision with some obstacles in the environment.
    %       (robot is currently in at least an obstacle's dange
    %       range)
    %   'WS': robot reaches its goal.
    %   'FS': robot collides with obstacles.
    %----------------------------------------------------------------------
    properties
        state_property
        state
        next_state
        next_state_property
        robot_pos
        orientation
        size1
        size2
        observe_range
        action_space
        action
        reward
        q_value
        detect_angles
        detect_range
        min_dist
        epsilon
        nn      % {layers, params, caches, grads}
        learning_rate
    end
    
    methods
        function obj = Robot(map_size, num_obs, sta_mov_split, epsilon, learning_rate, min_dist)
            %	Construct instances of this class
            
            obj@Map(map_size, num_obs, sta_mov_split);
            
            %   Safe state      : SS
            %   None-safe state : NS
            %   Winning state   : WS
            %   Failure state   : FS
            %------------------------------
            % Assume that start state is safe
            obj.state_property = 'SS';
            
            step = floor(obj.map_size/10);
            obj.size1 = 0.7*step;
            obj.size2 = step;
            
            obj.robot_pos = obj.start;
            obj.orientation = pi/4;
            obj.observe_range = floor(obj.map_size / 3);
            obj.mov_obs = obj.orig_mov_obs;
            
            obj.state = zeros(10, 1);
            obj.state(1:2) = obj.robot_pos';
            obj.state(3) = obj.orientation;
            obj.next_state = zeros(10, 1);
            
            obj.action_space = linspace(-pi/2, pi/2, 7);
            obj.action = 0;
            obj.reward = 0;
            obj.q_value = zeros(7, 1);
            
            obj.detect_angles = linspace(0, 2*pi, 8)';
            offset = pi/(length(obj.detect_angles) - 1);
            detect_angle1 = linspace(0, 2*pi, 8)';
            detect_angle2 = [detect_angle1(2:end); detect_angle1(1)];
            obj.detect_range = [detect_angle1 detect_angle2] - offset;
            
            obj.min_dist = min_dist;
            obj.epsilon = epsilon;
            
            % Setting the neural network
            n_x = length(obj.state);
            n_h = round(1.2 * n_x);
            n_y = length(obj.action_space);
            layers = struct('n_x', n_x, 'n_h', n_h, 'n_y', n_y);
            
            w1 = randn(n_h, n_x);
            b1 = zeros(n_h, 1);
            w2 = randn(n_y, n_h);
            b2 = zeros(n_y, 1);
            params = struct('w1', w1, 'b1', b1, 'w2', w2, 'b2', b2);
            
            z1 = zeros(n_h, 1);
            a1 = zeros(n_h, 1);
            z2 = zeros(n_y, 1);
            a2 = zeros(n_y, 1);
            caches = struct('z1', z1, 'a1', a1, 'z2', z2, 'a2', a2);
            
            dw1 = zeros(n_h, n_x);
            db1 = zeros(n_h, 1);
            dz1 = zeros(n_h, 1);
            da1 = zeros(n_h, 1);
            dw2 = zeros(n_y, n_h);
            db2 = zeros(n_y, 1);
            dz2 = zeros(n_y, 1);
            da2 = zeros(n_y, 1);
            grads = struct('dw1', dw1, 'db1', db1, 'dz1', dz1, 'da1', da1,...
                           'dw2', dw2, 'db2', db2, 'dz2', dz2, 'da2', da2);
            
            obj.nn = struct('layers', layers,...
                            'params', params,...
                            'caches', caches,...
                            'grads', grads);
            
            obj.learning_rate = learning_rate;
        end
        
        function D = read_distances(obj)
            D = calc_distances(obj.robot_pos, obj.obs);
            D(D > obj.observe_range) = 0;
        end
        
        function obj = observe_state(obj)
            D = obj.read_distances();
            angles = calc_angles(obj.robot_pos, obj.obs) - obj.orientation;

            for i = 1:length(obj.detect_angles)-1
                obj_idx = angles >= obj.detect_range(i, 1) &...
                          angles < obj.detect_range(i, 2);
                      
                % Retrieve distance signal from sensor
                detected = D(obj_idx);
                if max(detected) == 0
                    detected = 0;fprintf('1\n');
                else
                    detected(detected == 0) = Inf;
                    detected = min(detected);fprintf('2\n');
                end
                disp(obj.state(3+i));
                fprintf('\ndetected:');disp(size(detected));
                obj.state(3+i) = detected;
            end
        end
       
        function obj = get_state_property(obj)
            obs_with_goal = [obj.obs; obj.goal];
            D = calc_distances(obj.robot_pos, obs_with_goal);
            
            % Check for obstacles' danger ranges
            danger_idx = find(D < obj.danger_range);
            
            if isempty(danger_idx)
                obj.state_property = 'SS';
            elseif ~isempty(danger_idx)
                obj.state_property = 'NS';
            end
            if min((obj.robot_pos - obj.goal) < obj.min_dist) == 1
               obj.state_property = 'WS';
            elseif ~isempty(find(D < obj.min_dist, 1))
                obj.state_property = 'FS';
            end
        end
        
        function obj = get_next_state_property(obj)
            % Function gets the property from the next state
            next_obs_with_goal = [obj.next_obs; obj.goal];
            next_D = calc_distances(obj.next_state(1:2), next_obs_with_goal);
            
            % Check for obstacles' danger ranges
            danger_idx = find(next_D < obj.danger_range);
            
            if isempty(danger_idx)
                obj.next_state_property = 'SS';
            elseif ~isempty(danger_idx)
                obj.next_state_property = 'NS';
            end
            if min((obj.next_state(1:2) - obj.goal) < obj.min_dist) == 1
               obj.next_state_property = 'WS';
            elseif ~isempty(find(next_D < obj.min_dist, 1))
                obj.next_state_property = 'FS';
            end
        end
        
        function obj = get_reward(obj)
            if obj.state_property == 'NS' && obj.next_state_property == 'SS'
                obj.reward = obj.reward + 0.3;
            elseif obj.state_property == 'SS' && obj.next_state_property == 'NS'
                obj.reward = obj.reward - 0.2;
            elseif obj.state_property == 'NS' && obj.next_state_property == 'NS'
                D = max(obj.state(4:end));
                next_D = max(obj.next_state(4:end));
                if next_D > D
                    obj.reward = obj.reward + 0.4;
                else
                    obj.reward = obj.reward - 0.4;
                end
            elseif obj.state_property == 'WS'
                obj.reward = obj.reward + 1;
            elseif obj.state_property == 'FS'
                obj.reward = obj.reward - 0.6;
            end
        end
        
        function obj = get_q_value(obj)
            obj.nn.caches.z1 = obj.nn.params.w1*obj.state + obj.nn.params.b1;
            obj.nn.caches.a1 = sigmoid(obj.nn.caches.z1);
            obj.nn.caches.z2 = obj.nn.params.w2*obj.nn.caches.a1 + obj.nn.params.b2;
            obj.nn.caches.a2 = sigmoid(obj.nn.caches.z2);
            
            obj.q_value = obj.nn.caches.a2;
        end
        
        function obj = back_prob(obj)
           obj.nn.grads.da2 = 2*obj.nn.caches.a2 - 2*target;
           obj.nn.grads.dz2 = obj.nn.grads.da2 .* (obj.nn.caches.a2.*(1-obj.nn.caches.a2));
           obj.nn.grads.dw2 = obj.nn.grads.dz2 * obj.nn.caches.a1';
           obj.nn.grads.db2 = obj.nn.grads.dz2;
           
           obj.nn.grads.da1 = obj.nn.params.w2' * obj.nn.grads.dz2;
           obj.nn.grads.dz1 = obj.nn.grads.da1 .* (obj.nn.caches.a1.*(1-obj.nn.caches.a1));
           obj.nn.grads.dw1 = obj.nn.grads.dz1 * obj.state';
           obj.nn.grads.db1 = obj.nn.grads.dz1;
        end
        
        function obj = update_params(obj)
           obj.nn.params.w1 = obj.nn.params.w1 + obj.learning_rate*obj.nn.grads.dw1;
           obj.nn.params.b1 = obj.nn.params.b1 + obj.learning_rate*obj.nn.grads.db1;
           obj.nn.params.w2 = obj.nn.params.w2 + obj.learning_rate*obj.nn.grads.dw2;
           obj.nn.params.b2 = obj.nn.params.b2 + obj.learning_rate*obj.nn.grads.db2;
        end
        
        function obj = step(obj)
            % epsilon-Greedy
            prob = rand(1);
            if prob < obj.epsilon
                idx = randi([1 length(valid_Q)], 1);
            else
                [~, idx] = max(valid_Q);
            end
            
            % Observe the next position of robot
            obj.action = obj.action_space(idx);
            obj.next_state(3) = obj.orietation + obj.action;
            x = obj.robot_pos(1) + cosd(obj.next_state(3));
            y = obj.robot_pos(2) + sind(obj.next_state(3));
            obj.next_state(1:2) = [x; y];
            obj.get_state_property(obj);
            obj.get_next_state_property(obj);
            obj.get_reward(obj);
        end
        
    end
end

