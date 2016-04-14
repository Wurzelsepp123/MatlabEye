function [onset,offset]= getDirection(data,k)
if k<10
    disp('k too low');
    return;
end
i=1;
dx=zeros(3,1);
dy=zeros(3,1);
K=3; %max duration deviation
N=4; %max duraiton inconsistency
m=30; 
onset=1;
offset=100;

alpha=zeros(m,1);

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

% a, check deviation from main direction and inconsistency
% for i=1:size(alpha,1)
%     d(i)=abs(alpha(i)-dir);   
%     if (d(i)>60 && i<mid)
%         if mid-i < mid-onset
%             onset=i;
%         end
%     elseif (d(i)>60 && i>mid)
%         if i-mid < offset-mid
%             offset=i;
%         end
%     end
% end

d=abs(alpha-dir)>60;
d=double(d);

%get sum over three neighbouring values
tic
s=conv([1 1 1],d);
toc
s=s(3:size(s,1)-2);
% tic 
% for i=1:size(d)-2
%     s(i)=sum(d(i:i+2));
% end
% toc

%find closest consecutive deviation
ind=find(s==3);
dist_mid=(mid-ind);
dist_on=dist_mid(dist_mid>0);
%closest deviation to k where 3 samples are above thresh
if(size(dist_on)<1)
    disp('no onset found')
    return;
end
onset=dist_on(end)+2;

dist_off=dist_mid(dist_mid<0);
if(size(dist_off)<1)
    disp('no offset found')
    return;
end
offset=dist_off(1);


figure
plot(d)


%b,check for inconsitent direction
% tic
% for i=2:size(alpha,1)
%      e(i-1)=alpha(i)-alpha(i-1);
% end
% toc

e_conv=conv([1 -1],alpha);
e_conv=e_conv(2:end-1);
e_t=abs(e_conv)>=40;
e_t=double(e_t);
s4=conv([1 1 1 1],e_t);
s4=s4(4:size(s4,1)-3);

ind2=find(s4==4);
dist_mid2=(mid-ind2);
dist_on2=dist_mid2(dist_mid2>0);
onset2=dist_on2(end);

dist_off2=dist_mid2(dist_mid2<0);
offset2=dist_off2(1)-3;

onset=k-onset2;
offset=k-offset2;

end
