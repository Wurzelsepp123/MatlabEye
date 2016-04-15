function [onset,offset]= getOnOffset(data,k)
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
onset_found=false;
offset_found=false;
alpha=zeros(m,1);

%% a,deviatins calculate directions
for n=k-m:k+m
    dx(i)=(data(n+1,2)-data(n,2))/(data(n+1,1)-data(n,1));
    dy(i)=(data(n+1,3)-data(n,3))/(data(n+1,1)-data(n,1));    
    alpha(i)=atan(dy(i)/dx(i));      
    i=i+1;
end

alpha=rad2deg(alpha);
mid=ceil(size(alpha,1)/2);
dir=mean(alpha(mid-1:mid+1));

d=abs(alpha-dir)>60;
d=double(d);

%get sum over three neighbouring values

s_a=conv([1 1 1],d);
s_a=s_a(3:size(s_a,1)-2);


%find closest consecutive deviation
ind_a=find(s_a==3);
dist_mid_a=(mid-ind_a);
dist_on_a=dist_mid_a(dist_mid_a>0);
%closest deviation to k where 3 samples are above thresh
if(size(dist_on_a)>0)
    
    onset_a=dist_on_a(end)-2;
    onset_found=true;
else
    disp('no onset found by deviation');
end

dist_off_a=dist_mid_a(dist_mid_a<0);
if(size(dist_off_a)>0)
    offset_a=dist_off_a(1);
    offset_found=true;
    
else
    disp('no offset found by deviation')
    
end

if(onset_found && offset_found)
    onset=k-onset_a;
    offset=k-offset_a;
    return;
end


%% b,check for inconsitent direction
% tic
% for i=2:size(alpha,1)
%      e(i-1)=alpha(i)-alpha(i-1);
% end
% toc
onset_found=false;
offset_found=false;


e_conv=conv([1 -1],alpha);
e_conv=e_conv(2:end-1);
e_t=abs(e_conv)>=40;
e_t=double(e_t);
%find consecutive samples above threshold
s_b=conv([1 1 1 1],e_t);
s_b=s_b(4:size(s_b,1)-3);

ind_b=find(s_b==4);
dist_mid_b=(mid-ind_b);
dist_on_b=dist_mid_b(dist_mid_b>0);
if(size(dist_on_b)>0)
    onset_b=dist_on_b(end);
    onset_found=true;
end

dist_off_b=dist_mid_b(dist_mid_b<0);

if(size(dist_off_b)>0)
    offset_b=dist_off_b(1)-3;
    offset_found=true;
end


if(onset_found && offset_found)    
    onset=k-onset_b;
    offset=k-offset_b;
    
    onset_vel=(data(onset+1,2:3)-data(onset,2:3))/(data(onset+1,1)-data(onset,1));
    onset_vel2=dot(onset_vel,onset_vel);
    
    offset_vel=(data(offset+1,2:3)-data(offset,2:3))/(data(offset+1,1)-data(offset,1));
    offset_vel2=dot(offset_vel,offset_vel);
    
    
    peak_vel=(data(k+1,2:3)-data(k,2:3))/(data(k+1,1)-data(k,1));
    peak_vel2=dot(peak_vel,peak_vel);
    
    
    if (offset_vel2<0.2*peak_vel2 && onset_vel2<0.2*peak_vel2)
        disp('On/Offset Velocity below threshold !!');
        return;
    end
elseif onset_found || offset_found
    %%On or offset wasnt found
    if onset_found==false
        disp('Couldnt find onset!');
        onset=k-5;
        offset=k-offset_b;
    end
    if offset_found==false
        disp('Couldnt find offset!');
        offset=k+5;
        onset=k-onset_b;
    end
else
    disp('Couldnt find on and offset!');
    onset=k-5;
    offset=k+5;    
    
end

%% c, distance between dir change



end
