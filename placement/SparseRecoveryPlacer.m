classdef SparseRecoveryPlacer
	% Placer in huang2020sparse. Current version does not implement the
	% local cover part.
	
	properties(Constant)
		epsilon = 0.1;
		num_iter = 2;
	end
	
	methods(Static)
		function test()
			num_users = 5;
			radius = 20;
			area_side = 100;
			m_users = area_side*rand(num_users, 2);
			m_uavs = SparseRecoveryPlacer.place(m_users, radius)
		end
		
		function m_uavs = place(m_users,radius)
			num_users = size(m_users, 1);

			m_weights = ones(num_users, num_users);
			[m_uavs, m_bounds] = SparseRecoveryPlacer.solve_sparsity_program(m_users, m_weights, radius);

			for ind = 2:SparseRecoveryPlacer.num_iter
				m_weights = 1./(m_bounds + SparseRecoveryPlacer.epsilon);
				disp("SparseRecoveryPlacer iteration = " + num2str(ind))				
				[m_uavs, m_bounds] = SparseRecoveryPlacer.solve_sparsity_program(m_users, m_weights, radius);
				
			end
			
			m_uavs = SparseRecoveryPlacer.redundant_circle_deletion(m_users, m_uavs, radius);
			
		end
		
		function [m_uavs, m_bounds] = solve_sparsity_program(m_users, m_weights, radius)
			
			num_users = size(m_users, 1);
						
			cvx_begin quiet
			variable m_pos(num_users, 2)
			variable m_bounds(num_users, num_users)
			minimize( sum(sum(m_weights*m_bounds )) )
			subject to
			m_pos >= 0
			for ind_user_1=1:num_users
				for ind_user_2=1:num_users
					m_bounds(ind_user_1,ind_user_2) >= norm(m_pos(ind_user_1,:) - m_pos(ind_user_2,:))
				end
				norm(m_pos(ind_user_1,:) - m_users(ind_user_1,:)) <= radius
			end
			cvx_end
			
			m_uavs = m_pos;
		end
		
		function m_uavs = redundant_circle_deletion(m_users, m_uavs, radius)
			
			num_users = size(m_users, 1);
			num_uavs_now = size(m_uavs, 1);
			
			cover_indicators = zeros(num_users, num_uavs_now);
			for ind_user = 1:num_users
				for ind_uav_now = 1:num_uavs_now
					cover_indicators(ind_user, ind_uav_now) = (...
					    norm(m_users(ind_user,:) - m_uavs(ind_uav_now,:)) <= radius + SparseRecoveryPlacer.epsilon...
					);
				end
			end
			
			
			% Delete circles
			function [m_uavs, cover_indicators] = delete_circle_if_possible(m_uavs, cover_indicators)
				num_uavs_now = size(m_uavs,1);
				for ind_uav_now=1:num_uavs_now
					cols_remaining_uavs = [1:ind_uav_now-1,ind_uav_now+1:num_uavs_now];
					if all(cover_indicators <= sum(cover_indicators(:,cols_remaining_uavs),2))
						% delete uav `ind_uav`
						m_uavs = m_uavs(cols_remaining_uavs,:);
						cover_indicators=cover_indicators(:,cols_remaining_uavs);
						return
					end
				end
			end
			
			while 1
				[m_uavs_new, cover_indicators_new] = delete_circle_if_possible(m_uavs, cover_indicators);				
				if size(m_uavs_new,1) == size(m_uavs,1)
					break
				else
					m_uavs = m_uavs_new;
					cover_indicators = cover_indicators_new;
				end
					
			end
			
		end
	end
end

