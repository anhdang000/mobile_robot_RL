clc, clear, close all;

% Initialize
rbt = Robot(10, 5, 0.6, 0.12, 0.001, 0.2);
[fig, components] = rbt.init_figure();