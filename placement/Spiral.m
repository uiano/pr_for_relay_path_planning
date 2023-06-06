%created by Lyu Jiangbin on 2016-4-12

%Optimize the MBS placement with fixed locations for mobile terminals (MTs)
classdef Spiral
	
	methods(Static)
		
		function [m_uavs] = place(m_users, radius)
			% `m_users`: 2 x num_users
			%
			% `m_uavs`: 2 x num_uavs
			
			m_uavs=[];%location of existing MBS
			NextBo=[];
			IdxNextBoInKu=0;
			
			while ~isempty(m_users) %while Ku is not empty
				
				%recognize the boundary points
				Num_UnCoveredMT=size(m_users,2);
				IsBoundaryPoint=zeros(1,Num_UnCoveredMT);
				IdxKuInBo=zeros(1,Num_UnCoveredMT);
				if Num_UnCoveredMT>=3
					IdxBoundaryPointsCurve=convhull(m_users(1,:)',m_users(2,:)'); %boundary(Ku(1,:)',Ku(2,:)',0.0);%0.4 %the last parameter is the shrinking factor in range 0~1, 0 for convex hull, default 0.5
					IdxBoundaryPoints=IdxBoundaryPointsCurve;
					IdxBoundaryPoints(end)=[];%the last point is the same as the first point in boundary() return
					for i=1:length(IdxBoundaryPoints)
						IsBoundaryPoint(IdxBoundaryPoints(i))=1;
						IdxKuInBo(IdxBoundaryPoints(i))=i;
					end
				else
					IsBoundaryPoint=ones(1,Num_UnCoveredMT);
					IdxBoundaryPoints=1:Num_UnCoveredMT;
					IdxBoundaryPointsCurve=1:Num_UnCoveredMT;
				end
				
				
				
				IdxNextBo=find(IdxBoundaryPoints==IdxNextBoInKu);
				if isempty(IdxNextBo)
					IdxNextBo=1;
				end
				
				if isempty(NextBo)
					NextBo=m_users(:,IdxBoundaryPoints(1));
					IdxNextBo=1;
					IdxNextBoInKu=IdxBoundaryPoints(1);
				end
				
				
				BoIdx=find(IsBoundaryPoint);
				%InIdx=find(IsBoundaryPoint==0);
				
				IsCovered=zeros(1,Num_UnCoveredMT);
				IdxCovered=IdxNextBoInKu;
				IsCovered(IdxCovered)=1;
				
				IdxUncovered=find(IsCovered==0 & IsBoundaryPoint);
				MBSnew=NextBo;
				IsNextBoOnly=1;
				[IdxCovered, MBSnew, MBSnewIsChanged] = Spiral.LocalCover(m_users,IdxCovered,IdxUncovered, MBSnew, radius);
				IsCovered(IdxCovered)=1;
				if MBSnewIsChanged
					IsNextBoOnly=0;
				end
				
				IdxUncovered=find(IsCovered==0 & IsBoundaryPoint==0);
				[IdxCovered, MBSnew, MBSnewIsChanged] = Spiral.LocalCover(m_users,IdxCovered,IdxUncovered, MBSnew, radius);
				IsCovered(IdxCovered)=1;
				if MBSnewIsChanged
					IsNextBoOnly=0;
				end
				
				%%%ONLY for 80MTs plot, 11th MBS overlap with MT. Comment out the
				%%%following for other cases
				%     if IsNextBoOnly==1%only NextBo is covered
				%         MBSnew=MBSnew+[4;-35] %so that the location does not overlap in the plot
				%     end
				
				%returning MBSnew and IdxCovered
				
				
				NumNewlyCovered=length(IdxCovered);
				if NumNewlyCovered<Num_UnCoveredMT%%still there are MTs left
					
					NextBo=[];
					IdxNextBoInKu=0;
					K_bo=size(BoIdx,2);
					
					count=0;
					while count<K_bo
						if IdxNextBo+count<=K_bo
							Idx=IdxNextBo+count;
						else
							Idx=IdxNextBo+count-K_bo;
						end
						if IsCovered( IdxBoundaryPoints(Idx) )==0
							NextBo= m_users(:, IdxBoundaryPoints(Idx) );
							IdxNextBoInKu=IdxBoundaryPoints(Idx);
							break;
						end
						count=count+1;
					end
					
					temp=IdxNextBoInKu;
					for i=1:NumNewlyCovered
						if IdxCovered(i)<temp
							IdxNextBoInKu=IdxNextBoInKu-1;
						end
					end
					
				end
				
				m_users(:, IdxCovered)=[]; %update the uncovered MT set
				m_uavs=[m_uavs MBSnew]; %update the existing MTs
				
				if size(m_uavs,2)==1
					Ku1=m_users;
				elseif size(m_uavs,2)==2
					Ku2=m_users;
				end
				%     Num_UnCoveredMT=size(Ku,2);
				%     Me;
			end
			
		end
		
		
		function [IdxCovered, MBSnew, MBSnewIsChanged] = LocalCover(Ku,IdxCovered,IdxUncovered, MBSnew, r)
			MBSnewIsChanged=0;
			NumCovered=length(IdxCovered);
			NumCandidate=length(IdxUncovered);
			
			if NumCandidate==0
				[center,minmaxDistance] = Spiral.minboundcircle(Ku(1,[IdxCovered]),Ku(2,[IdxCovered]));
				u=center';
				MBSnew=u;
				MBSnewIsChanged=1;
				return;
			end
			
			while NumCandidate>0
				
				IdxCandidateDelete=[];
				for i=1:NumCandidate
					for j=1:NumCovered
						dist=norm(Ku(:,IdxUncovered(i))-Ku(:,IdxCovered(j)));
						if dist>2*r
							IdxCandidateDelete=[IdxCandidateDelete i];
							break;
						end
					end
				end
				
				IdxUncovered(IdxCandidateDelete)=[];
				NumCandidate=length(IdxUncovered);
				if NumCandidate==0
					[center,minmaxDistance] = Spiral.minboundcircle(Ku(1,[IdxCovered]),Ku(2,[IdxCovered]));
					u=center';
					MBSnew=u;
					MBSnewIsChanged=1;
					break;
				end
				
				U_Rep=repmat(MBSnew,1,NumCandidate); %repeat the vector u
				Difference=U_Rep-Ku(:,IdxUncovered);
				distance=sqrt(sum(Difference.^2,1));%The distance between u and the MTs
				IsCoveredCandidate= distance<=r*1.0;
				Newly_Covered_Idx_Candidate=find(IsCoveredCandidate);
				IdxCovered=[IdxCovered IdxUncovered(Newly_Covered_Idx_Candidate)];
				IdxUncovered(Newly_Covered_Idx_Candidate)=[];
				NumCandidate=length(IdxUncovered);
				if NumCandidate==0
					[center,minmaxDistance] = Spiral.minboundcircle(Ku(1,[IdxCovered]),Ku(2,[IdxCovered]));
					u=center';
					MBSnew=u;
					MBSnewIsChanged=1;
					break;
				end
				
				distance(Newly_Covered_Idx_Candidate)=[];
				[dis,IdxNearestOutlier]=min(distance);
				IdxNearestOutlierInKu=IdxUncovered(IdxNearestOutlier);
				
				[center,minmaxDistance] = Spiral.minboundcircle(Ku(1,[IdxCovered IdxNearestOutlierInKu]),Ku(2,[IdxCovered IdxNearestOutlierInKu]));
				u=center';
				
				if minmaxDistance<=r
					MBSnew=u;
					MBSnewIsChanged=1;
				else
					[center,minmaxDistance] = Spiral.minboundcircle(Ku(1,[IdxCovered]),Ku(2,[IdxCovered]));
					u=center';
					MBSnew=u;
					MBSnewIsChanged=1;
					break;
				end
				
			end
			%returning MBSnew and IdxCovered
			
		end
		
		function [center,radius] = minboundcircle(x,y,hullflag)
			% minboundcircle: Compute the minimum radius enclosing circle of a set of (x,y) pairs
			% usage: [center,radius] = minboundcircle(x,y,hullflag)
			%
			% arguments: (input)
			%  x,y - vectors of points, describing points in the plane as
			%        (x,y) pairs. x and y must be the same size. If x and y
			%        are arrays, they will be unrolled to vectors.
			%
			%  hullflag - boolean flag - allows the user to disable the
			%        call to convhulln. This will allow older releases of
			%        matlab to use this code, with a possible time penalty.
			%        It also allows minboundellipse to call this code
			%        efficiently.
			%
			%        hullflag = false --> do not use the convex hull
			%        hullflag = true  --> use the convex hull for speed
			%
			%        default: true
			%
			%
			% arguments: (output)
			%  center - 1x2 vector, contains the (x,y) coordinates of the
			%        center of the minimum radius enclosing circle
			%
			%  radius - scalar - denotes the radius of the minimum
			%        enclosing circle
			%
			%
			% Example usage:
			%   x = randn(50000,1);
			%   y = randn(50000,1);
			%   tic,[c,r] = minboundcircle(x,y);toc
			%
			%   Elapsed time is 0.171178 seconds.
			%
			%   c: [-0.2223 0.070526]
			%   r: 4.6358
			%
			%
			% See also: minboundrect
			%
			%
			% Author: John D'Errico
			% E-mail: woodchips@rochester.rr.com
			% Release: 1.0
			% Release date: 1/10/07
			
			% default for hullflag
			if (nargin<3) || isempty(hullflag)
				hullflag = true;
			elseif ~islogical(hullflag) && ~ismember(hullflag,[0 1])
				error 'hullflag must be true or false if provided'
			end
			
			% preprocess data
			x=x(:);
			y=y(:);
			
			% not many error checks to worry about
			n = length(x);
			if n~=length(y)
				error 'x and y must be the same sizes'
			end
			
			% start out with the convex hull of the points to
			% reduce the problem dramatically. Note that any
			% points in the interior of the convex hull are
			% never needed.
			if hullflag && (n>3)
				edges = convhulln([x,y]);
				
				% list of the unique points on the convex hull itself
				% convhulln returns them as edges
				edges = unique(edges(:));
				
				% exclude those points inside the hull as not relevant
				x = x(edges);
				y = y(edges);
				
			end
			
			% now we must find the enclosing circle of those that
			% remain.
			n = length(x);
			
			% special case small numbers of points. If we trip any
			% of these cases, then we are done, so return.
			switch n
				case 0
					% empty begets empty
					center = [];
					radius = [];
					return
				case 1
					% with one point, the center has radius zero
					center = [x,y];
					radius = 0;
					return
				case 2
					% only two points. center is at the midpoint
					center = [mean(x),mean(y)];
					radius = norm([x(1),y(1)] - center);
					return
				case 3
					% exactly 3 points
					[center,radius] = Spiral.enc3(x,y);
					return
			end
			
			% more than 3 points.
			
			% Use an active set strategy.
			aset = 1:3; % arbitrary, but quite adequate
			iset = 4:n;
			
			% pick a tolerance
			tol = 10*eps*(max(abs(mean(x) - x)) + max(abs(mean(y) - y)));
			
			% Keep a list of old sets as tried to trap any cycles. we don't need to
			% retain a huge list of sets, but only a few of the best ones. Any cycle
			% must hit one of these sets. Really, I could have used a smaller list,
			% but this is a small enough size that who cares? Almost always we will
			% never even fill up this list anyway.
			old.sets = NaN(10,3);
			old.rads = inf(10,1);
			old.centers = NaN(10,2);
			
			flag = true;
			while flag
				% have we seen this set before? If so, then we have entered a cycle
				aset = sort(aset);
				if ismember(aset,old.sets,'rows')
					% we have seen it before, so trap out
					center = old.centers(1,:);
					radius = old.rads(1);
					
					% just reset flag then continue, and the while loop will terminate
					flag = false;
					continue
				end
				
				% get the enclosing circle for the current set
				[center,radius] = Spiral.enc3(x(aset),y(aset));
				
				% is this better than something from the retained sets?
				if radius < old.rads(end)
					old.sets(end,:) = sort(aset);
					old.rads(end) = radius;
					old.centers(end,:) = center;
					
					% sort them in increasing order of the circle radii
					[old.rads,tags] = sort(old.rads,'ascend');
					old.sets = old.sets(tags,:);
					old.centers = old.centers(tags,:);
				end
				
				% are all the inactive set points inside the circle?
				r = sqrt((x(iset) - center(1)).^2 + (y(iset) - center(2)).^2);
				[rmax,k] = max(r);
				if (rmax - radius) <= tol
					% the active set enclosing circle also enclosed
					% all of the inactive points.
					flag = false;
				else
					% it must be true that we can replace one member of aset
					% with iset(k). Which one?
					s1 = [aset([2 3]),iset(k)];
					[c1,r1] = Spiral.enc3(x(s1),y(s1));
					if (norm(c1 - [x(aset(1)),y(aset(1))]) <= r1)
						center = c1;
						radius = r1;
						
						% update the active/inactive sets
						swap = aset(1);
						aset = [iset(k),aset([2 3])];
						iset(k) = swap;
						
						% bounce out to the while loop
						continue
					end
					s1 = [aset([1 3]),iset(k)];
					[c1,r1] = Spiral.enc3(x(s1),y(s1));
					if (norm(c1 - [x(aset(2)),y(aset(2))]) <= r1)
						center = c1;
						radius = r1;
						
						% update the active/inactive sets
						swap = aset(2);
						aset = [iset(k),aset([1 3])];
						iset(k) = swap;
						
						% bounce out to the while loop
						continue
					end
					s1 = [aset([1 2]),iset(k)];
					[c1,r1] = Spiral.enc3(x(s1),y(s1));
					if (norm(c1 - [x(aset(3)),y(aset(3))]) <= r1)
						center = c1;
						radius = r1;
						
						% update the active/inactive sets
						swap = aset(3);
						aset = [iset(k),aset([1 2])];
						iset(k) = swap;
						
						% bounce out to the while loop
						continue
					end
					
					% if we get through to this point, then something went wrong.
					% Active set problem. Increase tol, then try again.
					tol = 2*tol;
					
				end
				
			end
		end
		% =======================================
		%  begin subfunction
		% =======================================
		function [center,radius] = enc3(X,Y)
			% minimum radius enclosing circle for exactly 3 points
			%
			% x, y are 3x1 vectors
			
			% convert to complex
			xy = X + sqrt(-1)*Y;
			
			% just in case the points are collinear or nearly so, get
			% the interpoint distances, and test the farthest pair
			% to see if they work.
			Dij = @(XY,i,j) abs(XY(i) - XY(j));
			D12 = Dij(xy,1,2);
			D13 = Dij(xy,1,3);
			D23 = Dij(xy,2,3);
			
			% Find the most distant pair. Test if their circumcircle
			% also encloses the third point.
			if (D12>=D13) && (D12>=D23)
				center = (xy(1) + xy(2))/2;
				radius = D12/2;
				if abs(center - xy(3)) <= radius
					center = [real(center),imag(center)];
					return
				end
			elseif (D13>=D12) && (D13>=D23)
				center = (xy(1) + xy(3))/2;
				radius = D13/2;
				if abs(center - xy(2)) <= radius
					center = [real(center),imag(center)];
					return
				end
			elseif (D23>=D12) && (D23>=D13)
				center = (xy(2) + xy(3))/2;
				radius = D23/2;
				if abs(center - xy(1)) <= radius
					center = [real(center),imag(center)];
					return
				end
			end
			
			% if we drop down to here, then the points cannot
			% be collinear, so the resulting 2x2 linear system
			% of equations will not be singular.
			A = 2*[X(2)-X(1), Y(2)-Y(1); X(3)-X(1), Y(3)-Y(1)];
			rhs = [X(2)^2 - X(1)^2 + Y(2)^2 - Y(1)^2; ...
				X(3)^2 - X(1)^2 + Y(3)^2 - Y(1)^2];
			
			center = (A\rhs)';
			radius = norm(center - [X(1),Y(1)]);
			
			
		end
		
		function test()
			b_load = 1;
			if b_load
				load('Spiral-test.mat')
			else				
				radius = 20;
				m_users = 80*rand(2,100);
			end
			
			m_uavs = Spiral.test_placement(m_users, radius);
					
			for ind_user=1:size(m_users,2)
				dist = Spiral.distance_to_closest_uav(m_uavs, m_users(:,ind_user));
				if dist>radius
					disp(dist)
				end
			end
		end
		
		function d = distance_to_closest_uav(m_uavs, user_coords)
			dif = m_uavs - repmat(user_coords(:),[1,size(m_uavs,2)]);
			dists = sqrt(sum(dif.^2,1));
			d = min(dists);
			
		end
		
		
		function m_uavs = test_placement(m_users, radius)
			clc;
			%clear all;
			close all;
			
			% %R=200; %The communication range in meters between any pair of MBSs
			%
			% radius=100; %The communication range in meters between a MBS and a MT
			%
			% rho=400% MT density in MTs/km^2;
			% num_MTs=100; %total number of MTs
			%
			% area=num_MTs/rho%The area in square km
			% len_area=sqrt(area)*1000; %The side length of the square region where the MTs are located
			%
			% x_coordinate=len_area*rand(1,num_MTs)-len_area/2; %x coordinate
			% y_coordinate=len_area*rand(1,num_MTs)-len_area/2; %y cordinate
			%
			% load('Example80.mat','x_coordinate','y_coordinate','R','r','rho','K','A','L');
			%
			% x_coordinate = 600*rand(1,50) - 300;
			% y_coordinate = 600*rand(1,50) - 300;
			
			
			[m_uavs] = Spiral.place(m_users, radius);
			
			x_coordinate = m_users(1,:);
			y_coordinate = m_users(2,:);
			
			DivideBy=1; %L*0.1581*2;
			len_area =1;
			
			angle_vec=0:0.01:2*pi;
			x_incr=radius/DivideBy*cos(angle_vec);
			y_incr=radius/DivideBy*sin(angle_vec);
			
			figure;
			h1=plot(x_coordinate/DivideBy+len_area/2/DivideBy, y_coordinate/DivideBy+len_area/2/DivideBy, '^k','MarkerFaceColor',[0,1,1]);
			hold on;
			%xlim([0, L/DivideBy]);
			%ylim([0,L/DivideBy]);
			
			M=size(m_uavs,2)
			for idx=1:M
				x0=m_uavs(1,idx)/DivideBy+len_area/2/DivideBy;
				y0=m_uavs(2,idx)/DivideBy+len_area/2/DivideBy;
				h2=plot(x0,y0,'s', 'MarkerSize',10,'MarkerEdgeColor','k', 'MarkerFaceColor',[.49 1 .63]);%
				plot(2*len_area,2*len_area,'x', 'MarkerSize',10,'MarkerEdgeColor','k', 'MarkerFaceColor',[.49 1 .63],'linewidth',2);%just for legend
				plot(2*len_area,2*len_area,'+', 'MarkerSize',10,'MarkerEdgeColor','k', 'MarkerFaceColor',[.49 1 .63],'linewidth',1);%just for legend
				
				hold on;
				text(x0+5/DivideBy,y0,sprintf('%d',idx),'FontWeight','bold','FontSize',18);
				hold on;
				%         if(idx>=2)
				%             for i=1:(idx-1)
				%                if norm(Me(:,idx)-Me(:,i),2) <=R
				%                    h3=plot([Me(1,i),x0],[Me(2,i),y0], '-r','LineWidth',5);
				%                    hold on;
				%                end
				%             end
				%         end
				
				if idx<M
					x1=m_uavs(1,idx+1)/DivideBy+len_area/2/DivideBy;
					y1=m_uavs(2,idx+1)/DivideBy+len_area/2/DivideBy;
					hold on;
					arrow3([x0 y0],[x1 y1],'-.r',0.5,1.5);
				end
				
				hold on;
				x_vec=x0+x_incr;
				y_vec=y0+y_incr;
				plot(x_vec, y_vec, '--g','linewidth',2);
			end
			
			xlabel('x (km)','Fontsize',30);
			ylabel('y (km)','Fontsize',30);
			set(gcf,'Color','w');
			set(gca,'FontSize',15);
			
		end
		
		function out = save(m_users, radius)
			
			save('Spiral-test.mat','m_users','radius');
			out = ''
		end
		
		function load_and_test()
			load('Spiral-test.mat')
			
			
		end
	end
end