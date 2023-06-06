function limited_backhaul

experiment_1();

end

function experiment_1()
% Checks for one of the subproblems of rate allocation with constrained
% backhaul. notes-cartography.pdf 2021/08/01.

M = 10;
v_y = randn(M,1);
rho = 0.5;
rmax = 3; 

% CVX 
cvx_begin
variable v_x(M)
variable s
minimize( s + (rho/2)*sum_square(v_x - v_y) )
subject to
sum(v_x) == rmax
v_x <= s
cvx_end

% Optimality conds
mu = (-rho*rmax + rho*sum(v_y) - 1)/M;

%res1 = rmax - sum(min( s*ones(M,1) ,  v_y - mu/rho*ones(M,1)  ))
res1 = residual_min_max_conds_option_1(v_y, mu, rho, rmax, s)
res2 = residual_min_max_conds_option_2(v_y, mu, rho, rmax, s)

opt_cond_s = norm(v_x - min( s*ones(M,1) ,  v_y - mu/rho*ones(M,1)  ))

% Residual vs. s
s_pts = linspace(-1, 2, 100);
for ind_pt=length(s_pts):-1:1
	res(ind_pt) = residual_min_max_conds_option_1(v_y, mu, rho, rmax, s_pts(ind_pt));
end
plot(s_pts, res);
grid on
end

function res = residual_min_max_conds_option_1(v_y, mu, rho, rmax, s)
M = size(v_y,1);
res = rmax - sum(min( s*ones(M,1) ,  v_y - mu/rho*ones(M,1)  ));
end

function res = residual_min_max_conds_option_2(v_y, mu, rho, rmax, s)
M = size(v_y,1);
res = sum(max( rho*(v_y - s*ones(M,1)) ,  mu ))-mu*M -1;
end



function experiment_2()
% Checks for one of the subproblems of rate allocation with constrained
% backhaul. notes-cartography.pdf 2021/08/01.

M = 10;
v_y = randn(M,1);
rmin = 3; 
v_c = rand(M,1);

% CVX 
cvx_begin
variable v_z(M)
dual variable lambda_cvx
minimize( (1/2)*sum_square(v_z - v_y) )
subject to
lambda_cvx: sum(v_z) == rmin
v_z >= 0
v_z <= v_c
cvx_end

% Optimality conds
lambda_cvx = -lambda_cvx
res_cvx = sum( min(max(0, v_y-lambda_cvx),v_c)) - rmin
lambda_cvx

lambda_conds = fsolve(@(x) residual_pseudo_simplex(v_y, x, v_c, rmin), 0)

% Residual vs. s
lambda_pts = linspace(-1, 2, 100);
for ind_pt=length(lambda_pts):-1:1
	res(ind_pt) = residual_pseudo_simplex(v_y, lambda_pts(ind_pt), v_c, rmin);
end
plot(lambda_pts, res);
grid on
end

function res = residual_pseudo_simplex(v_y, lambda, v_c, rmin)
M = size(v_y,1);
res = sum( min(max(0, v_y-lambda),v_c)) - rmin;

end
