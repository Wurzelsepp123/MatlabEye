function [onset,offset]= getDirection(data,k)
if k<4
    disp('k too low');
    return;
end
i=1;
dx=zeros(3,1);
dy=zeros(3,1);
alpha=zeros(3,1);
m=4;
onset=1;
offset=100;
%calculate directions
for n=k-m:k+m
    dx(i)=(data(n+1,2)-data(n,2))/(data(n+1,1)-data(n,1));
    dy(i)=(data(n+1,3)-data(n,3))/(data(n+1,1)-data(n,1));    
    alpha(i)=atan(dy(i)/dx(i));  
    
    i=i+1;
end
 alpha=rad2deg(alpha);
mid=ceil(size(alpha,1)/2);
dir=mean(alpha(mid-1:mid+1));

% check deviation from main direction and inconsistency
for i=1:size(alpha,1)
    d(i)=abs(alpha(i)-dir);   
    if (d(i)>60 && i<mid)
        if mid-i < mid-onset
            onset=i;
        end
    elseif (d(i)>60 && i>mid)
        if i-mid < offset-mid
            offset=i;
        end
    end
end
figure
plot(d)
hold on
plot(1:size(alpha,1), 60*ones(1,size(alpha,1)), 'LineWidth', 5);

%check for inconsitent direction
for i=2:size(alpha,1)
     e(i)=alpha(i)-alpha(i-1);
end
abs(e)>=40;


hold off
end
