
classdef sparsity
	properties(Constant)
		fi_data = "output/capacity_test.mat"
	end

	methods
		function obj = sparsity(obj)
			disp("hello")
			
			load(obj.fi_data)
			
			num_const = size(E,1);
			num_vars = size(E,2);
			num_groups = num_vars/num_vars_per_group;
			
			%Ex = E*diag(1+1e-4*randn(1,num_vars));
			Ex = E;
			X_out = obj.sparse_placement(Ex,f,num_vars_per_group,C);
						
			X_cand = zeros(num_vars_per_group, num_groups);
			X_cand(:,17) = 3;
			X_cand(4,17) = 2.5;
			X_cand(4,152) = 0.5;
			
			disp("CVX")
			sparsity.check_sol(X_out, Ex, f)
			disp("Cand")
			sparsity.check_sol(X_cand, Ex, f)
		end
		
	end
	
	methods(Static)
		function out = save_data(E, f, num_vars_per_group, C)
			out = pwd();
			save(sparsity.fi_data, 'E', 'f', 'num_vars_per_group', 'C');						
		end
		
		
		function X_out = sparse_placement(E,f, num_vars_per_group, C)
						
			num_const = size(E,1);
			num_vars = size(E,2);
			num_groups = num_vars/num_vars_per_group;
			
			mode = "group";
			if strcmp(mode, "l1")
				% num_vars_per_group is required, not C
				cvx_begin
				variable x(num_vars)
				minimize( norm(x, 1) )
				subject to
				Ex*x >= f
				cvx_end
				
				if strcmp(cvx_status,'Solved')
					X_out = reshape(x,num_vars_per_group,num_groups);
				else
					X_out = NaN;
				end
			elseif strcmp(mode, "group")
				
				cvx_begin
				variable s(num_groups,1)
				variable X(num_vars_per_group, num_groups)
				minimize(sum(s))
				subject to
				E*vec(X) >= f				
				vec(X) <= kron(eye(num_groups), ones(num_vars_per_group,1))*s
				cvx_end
				
				if strcmp(cvx_status,'Solved')
					X_out = X;
				else
					X_out = NaN;
				end
							
			end			
		end
	
		
		function check_sol(Sol, Ex, f)
			negative_part = sum(Sol(Sol(:)<0))
			residual = Ex*Sol(:) - f;
			neg_residual_part = norm(residual(residual<0))
			objective = sum(max(Sol,[],1))
			
		end
		
	end
	
	% Used to test the ADMM algorithm for SparsePlacer
	methods(Static)
			
		function m_R = x_step(v_weights, m_Z, m_U, admm_step)
			
% 			if 0
% 				save(sparsity.fi_data,'v_weights', 'm_Z', 'm_U', 'admm_step')
% 			else
% 				load(sparsity.fi_data)
% 			end
			
			num_gridpts = size(m_Z, 2)
			num_users = size(m_Z, 1)
			cvx_begin
			variable v_s(num_gridpts,1)
			variable m_R(num_users, num_gridpts)
			minimize( v_weights'*v_s + admm_step/2*sum_square(vec(m_R - m_Z + m_U)))
			subject to
			m_R <= ones(num_users,1) *v_s'
			cvx_end
			
			
		end
		
		function m_Z = z_step(min_user_rate, m_capacity, m_R, m_U)
			
% 			if 0
% 				save(sparsity.fi_data,'min_user_rate', 'm_capacity', 'm_R', 'm_U')
% 				return
% 			else
% 				load(sparsity.fi_data)
% 			end
			
			num_gridpts = size(m_R, 2);
			num_users = size(m_R, 1);
			cvx_begin			
			variable m_Z(num_users, num_gridpts)
			minimize( sum_square(vec(m_R - m_Z + m_U)))
			subject to
			m_Z * ones(num_gridpts,1) == min_user_rate*ones(num_users,1)
			0<= m_Z
			m_Z <= m_capacity
			cvx_end
			
			
		end
		
	end
	
end