classdef Map
    % The environment
    
    properties
        map_size
        step
        num_obs
        num_sta_obs
        num_mov_obs
        border
        sta_obs
        orig_mov_obs
        mov_obs
        obs
        next_mov_obs
        next_obs
        start
        goal
        danger_range
    end
    
    methods
        function obj = Map(map_size, num_obs, sta_mov_split)
            % Construct instances of this class
            
            obj.map_size = map_size;
            obj.step = floor(obj.map_size/10);
            obj.num_obs = num_obs;
            obj.num_sta_obs = round(obj.num_obs*sta_mov_split);
            obj.num_mov_obs = obj.num_obs - obj.num_sta_obs;
            obj.border = floor(obj.map_size / 5);
            obj.sta_obs = randi([obj.border map_size-obj.border], obj.num_sta_obs, 2);
            obj.orig_mov_obs = randi([obj.border map_size-obj.border], obj.num_mov_obs, 2);
            obj.mov_obs = obj.orig_mov_obs;
            obj.obs = [obj.sta_obs; obj.mov_obs];
            obj.next_mov_obs = obj.mov_obs;
            obj.next_obs = [obj.sta_obs; obj.next_mov_obs];
            obj.start = [obj.border obj.border];
            obj.goal = [map_size-obj.border map_size-obj.border];
            obj.danger_range = floor(map_size / 5);
        end
        
        function is_valid = check_valid(obj)
            is_valid = obj.next_obs > 0 & obj.next_obs < obj.map_size;
            is_valid = is_valid(:, 1) & is_valid(:, 2);
        end
        
        function obj = observe_obs_mov(obj, step)
            % Observe next obstacles' positions
            % according to their random moves
            
            is_valid = ones(obj.num_obs, 1);
            while is_valid
                obj.next_mov_obs(:, 1) = obj.mov_obs(:, 1) + step * randn(obj.num_mov_obs, 1);
                obj.next_mov_obs(:, 2) = obj.mov_obs(:, 2) + step * randn(obj.num_mov_obs, 1);
                is_valid = check_valid(obj);
            end
            obj.next_obs = [obj.sta_obs; obj.next_mov_obs];
        end
    end
end

