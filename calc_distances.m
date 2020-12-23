function D = calc_distances(o, A)

D = sqrt((A(:, 1) - o(1))^2 + (A(:, 2) - o(2))^2);
end

