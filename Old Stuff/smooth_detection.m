function smooth_detection(data,user,cluster)
close all;
%Check data for errors and remove them if selected
if mean(data{user,cluster}(:,4))~=1;
    disp('Data contains measuring errors!')
    a = input('Remove measuring errors (y/n)? ','s');
    if strcmpi(a,'y');
        data{user,cluster}=data{user,cluster}(data{user,cluster}(:,4)==1,:);
    end
%     if strcmpi(a,'n');
%         return;
%     end
end



%% Construct Filter
%     % Filter
%     fpass = 0.01;       % ]0;1[: pass frequency (for further explanation: "doc designfilt")
%     fstop = 0.05;       % ]0;1[: stop frequency
%     apass = 1;          % [dB]: pass attenuation
%     astop = 15;         % [dB]: stop attenuation
%
% filterr = designfilt('lowpassfir','PassbandFrequency',fpass,...
%     'StopbandFrequency',fstop,'PassbandRipple',apass,'StopbandAttenuation',astop);
% filter_coeff = filterr.Coefficients;
% %fvtool(filterr)
% order=filtord(filterr);
% 
% tmp = conv(data_train{i,j}{k}(l:l+sequencelength-1,2)',filter_coeff);
%                 tmp = (1-2*d/h*tand(tmp))*g_h/2;
% 


use_filter=true;
if use_filter
%     data_filtered=[data{user,cluster}(:,1),filter2(fspecial('gaussian',2),data{user,cluster}(:,2:3))];    
    x_smooth=smooth(data{user,cluster}(:,1),data{user,cluster}(:,2));
    y_smooth=smooth(data{user,cluster}(:,1),data{user,cluster}(:,3));
%      data_d=downsample([data{user,cluster}(:,1),x_smooth,y_smooth],15);
     data_d=downsample([data{user,cluster}(:,1),x_smooth,y_smooth],2);
else
%      data_d=downsample(data{user,cluster},15);
     data_d=downsample(data{user,cluster},2);
end


%% Plot filtered data
% figure
% scatter3(data_filtered(:,1),data_filtered(:,2),data_filtered(:,3),10,'r')
% xlabel('Time'),ylabel('X'),zlabel('Y')
% hold on
% scatter3(data{user,cluster}(:,1),data{user,cluster}(:,2),data{user,cluster}(:,3),10,'b')
% hold off
%%

%data_d =cellfun(@(x) downsample(x,15),data,'UniformOutput',false); % only take every 16. value: 1000Hz->62.5Hz
data_d(:,5)=0;

window_size=15;
teta=-pi:0.01:pi;
k=1;

max_smooth2=0.0625; %15°
min_smooth2=15e-4;%~2,4°

max_v_sp2=100^2;
min_v_sp2=3^2;

%min_smooth2=0.0064 %~5° bad detection for smooth persuit 
min_saccade2=2.56; %100°
num_windows=size(data_d,1)-window_size+1;
avg_type=zeros(num_windows,5);
saccade=false;
d=zeros(14,2);
%max_dist=0;
n=1;

%% Acceleration based saccade detection(nyström)
%Saccade and PSO threshold
tx=2.3e4;
ty=4.2e4;
t_min=20; %minimum time between two saccades in ms

%calculate acceleration--> d/t^2
d_data=diff(data_d(:,1:3));
v_xy=bsxfun(@rdivide,d_data(:,2:3),d_data(:,1));
v_x=v_xy(:,1);
v_y=v_xy(:,2);

a_xy=bsxfun(@rdivide,v_xy,d_data(:,1));
a_x=a_xy(:,1);
a_y=a_xy(:,2);

%stores saccade position
s_pos=zeros(size(a_x,1),1);

%calculate saccade threshold (400ms =200samples) and mark saccades
for i=1:size(a_x,1)
    %use default values
    if i<=200
        if (abs(a_x(i))>tx || abs(a_y(i))>ty)
            s_pos(i)=1;
        end
    %calculate new treshold based on last samples
    elseif i>200
        tx=6*std(a_x(i-200:i-1));
        ty=6*std(a_y(i-200:i-1));
        if (abs(a_x(i))>tx || abs(a_y(i))>ty)
            s_pos(i)=1;
        end
    end
end

%check if time between two saccades >t_min
j=0;
for i=1:size(s_pos,1)
    if s_pos(i)
        while j<t_min/2 && i+j<size(s_pos,1)
            if s_pos(i+j)
                s_pos(i:i+j)=1;   
                i=i+j+1;
                break;
            end
            j=j+1;
        end
        j=0;
    end
end

%Plots
figure;
plot(a_y);
hold on
plot(v_y*1000);
hold off
figure
c=bsxfun(@times,s_pos,[1 0 0]);
scatter(1:size(a_y),a_y,10,c);
hold on
plot(1:size(a_y,1), ty*ones(1,size(a_y,1)), 'LineWidth', 5);
plot(1:size(a_y,1), -ty*ones(1,size(a_y,1)), 'LineWidth', 5);
hold off

%% saccade main direction (for on/offset detection)

%find maximum peak of each saccade
n=0;
k=1;
s_index=find(s_pos);
for i=2:size(s_index)
    if s_index(i)==s_index(i-1)+1
        n=n+1;
    else
        [M,I]=max(abs(v_y(s_index(i-(n+1)):s_index(i-1))));
        max_ind(k)=s_index(i-(n+1))+I-1;
        k=k+1;
        n=0;
    end
    %if last s_index element is a seperate peak
    if (i==size(s_index,1) && n==0)
        max_ind(k)=s_index(i);
        k=k+1;
    end
end

for i=1:size(max_ind,1)
    [onset(i),offset(i)]=getDirection(data_d,max_ind(i));
end


while(k<=num_windows)
    
    
    window=data_d(k:k+window_size-1,1:3);
    %angular distance and saccade detection  
    d=diff(window(:,2:3));
    d_new=diff(window(:,1:3));
    %divide angular distance by time interval in case samples are cut
    %out(saccades/measurement errors)
    v=bsxfun(@rdivide,d_new(:,2:3),d_new(:,1));
%     if sum(dot(d,d,2) > 0.64)        
%         %disp('\n Saccade detected!')
%         saccade_log(n)=max(dot(d(:,:),d(:,:),2));
%         n=n+1;
%         saccade=true;       
%     end
    if sum(dot(v,v,2) > 100^2)        
%         disp('\n Saccade detected!')        
        n=n+1;
        saccade=true;       
    end

    
    % if saccade==false
    
    %median
    %average_distance=mean(d);
    avg_v=mean(v);
    
    %%
    %Visualization
    %         scatter3(window(:,1),window(:,2),window(:,3));
    %         xlabel('Time'),ylabel('X'),zlabel('Y')
    %        % 2.4-7.2 °/window -->0.16-0.48 °/frame
    %         figure
    %         scatter(d(:,1),d(:,2));
    %         axis equal
    %         hold on
    %         scatter(average_distance(1),average_distance(2));
    %         r=0.23;
    %         x_c=r*cos(teta);
    %         y_c=r*sin(teta);
    %         plot(average_distance(1)+x_c,average_distance(2)+y_c);
    %
    %         r=2.6e-4;
    %         x_c=r*cos(teta);
    %         y_c=r*sin(teta);
    %         plot(average_distance(1)+x_c,average_distance(2)+y_c);
    %%
    
    %mean
    %dist2=dot(average_distance,average_distance);
    avg_v2=dot(avg_v,avg_v);
    
%     if dist2<max_smooth2 && dist2>min_smooth2
%         if average_distance(2)<0
%             %disp('\n Smooth Pursuit Down!!!!');
%             avg_type(k,:)=[average_distance,[0 1 0]];
%             data_d(k:k+window_size-1,5)=1;
%         elseif average_distance(2)>0
%             %disp('\n Smooth Pursuit Up!!!!');
%             avg_type(k,:)=[average_distance,[0 0 1]];
%             data_d(k:k+window_size-1,5)=2;
%         end
%     elseif dist2 <min_smooth2
%         %             if dist2>max_dist
%         %                 max_dist=dist2;
%         %             end
%         %disp('\n Fixation!!!!');
%         avg_type(k,:)=[average_distance,[1 0 0]];
%         if data_d(k:k+window_size-1,5)~=1
%             data_d(k:k+window_size-1,5)=3;
%         end
%     else
%         %disp('\n Saccade? or whatever');
%         avg_type(k,:)=[average_distance,[0 0 0]];
%     end

if avg_v2<max_v_sp2 && avg_v2>min_v_sp2
    if avg_v(2)<0
        %disp('\n Smooth Pursuit Down!!!!');
        avg_type(k,:)=[avg_v,[0 1 0]];
        data_d(k:k+window_size-1,5)=1;
    elseif avg_v(2)>0
        %disp('\n Smooth Pursuit Up!!!!');
        avg_type(k,:)=[avg_v,[0 0 1]];
        data_d(k:k+window_size-1,5)=2;
    end
elseif avg_v2 <min_v_sp2
    %             if dist2>max_dist
    %                 max_dist=dist2;
    %             end
    %disp('\n Fixation!!!!');
    avg_type(k,:)=[avg_v,[1 0 0]];
    if data_d(k:k+window_size-1,5)~=1
        data_d(k:k+window_size-1,5)=3;
    end
else
    %disp('\n Saccade? or whatever');
    avg_type(k,:)=[avg_v,[0 0 0]];
end

% end
    k=k+1;
end

%Plot average distance distribution
figure
scatter(avg_type(:,1),avg_type(:,2),1,avg_type(:,3:5));
% hold on
% r=sqrt(max_smooth2);
% x_c=r*cos(teta);
% y_c=r*sin(teta);
% plot(x_c,y_c);
% 
% r=sqrt(min_smooth2);
% x_c=r*cos(teta);
% y_c=r*sin(teta);
% plot(x_c,y_c);
% 
% r=sqrt(0.0064);
% x_c=r*cos(teta);
% y_c=r*sin(teta);
% plot(x_c,y_c);
% hold off

hold on
r=sqrt(max_v_sp2);
x_c=r*cos(teta);
y_c=r*sin(teta);
plot(x_c,y_c);

r=sqrt(min_v_sp2);
x_c=r*cos(teta);
y_c=r*sin(teta);
plot(x_c,y_c);

% r=sqrt(0.0064);
% x_c=r*cos(teta);
% y_c=r*sin(teta);
% plot(x_c,y_c);
hold off

%Plot input data with assigned activity
color=zeros(size(data_d(:,1),1),3);
for i=1:size(data_d(:,1),1)
    switch data_d(i,5)
        case 0
            color(i,:)=[0 0 0];
        case 1
            color(i,:)=[0 1 0];
        case 2
            color(i,:)=[0 0 1];
        case 3
            color(i,:)=[1 0 0];
    end
    
end
figure
scatter3(data_d(:,1),data_d(:,2),data_d(:,3),10,color)
xlabel('Time'),ylabel('X'),zlabel('Y')
hold on
    scatter3(data_d(find(s_pos),1),data_d(find(s_pos),2),data_d(find(s_pos),3),15,'k')
hold off

if saccade
    disp('Data contains saccade')
end

if mean(data_d(:,5)==1)>0.8
    disp('Data is smooth persuit down')
elseif mean(data_d(:,5)==2)>0.8
    disp('Data is smooth persuit up')
elseif mean(data_d(:,5)==3)>0.8
    disp('Data is fixation');
else
    disp('Data type was not clearly detected')
end


%export log data to workspace
if exist('saccade_log')
    assignin('base', 'saccade_l', saccade_log)
end


end