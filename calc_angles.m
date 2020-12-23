function angles = calc_angles(o, A)

angles = atan((A(:, 2) - o(2))/(A(:, 1) - o(1)));
end

