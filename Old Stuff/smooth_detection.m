function smooth_detection(data,user,cluster)
if mean(data{user,cluster}(:,4))~=1;
    disp('Data contains measurement errors!')
    a = input('Use this data anyway (y/n)? ','s');
    if strcmpi(a,'n');
        return;
    end
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

use_filter=false;
if use_filter
    data_filtered=[data{user,cluster}(:,1),filter2(fspecial('gaussian',2),data{user,cluster}(:,2:3))];
    data_d=downsample(data_filtered,15);
else
    data_d=downsample(data{user,cluster},15);
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
min_saccade2=2.56; %100°
num_windows=size(data_d,1)-window_size+1;
avg_type=zeros(num_windows,5);
saccade=false;
d=zeros(14,2);
%max_dist=0;
n=1;

while(k<=num_windows)
    
    
    window=data_d(k:k+window_size-1,1:3);
    %angular distance and saccade detection        
    d=diff(window(:,2:3));    
    if sum(dot(d(:,:),d(:,:),2) > 0.64)        
        %disp('\n Saccade detected!')
        saccade_log(n)=max(dot(d(:,:),d(:,:),2));
        n=n+1;
        saccade=true;       
    end
    
    % if saccade==false
    
    %median
    average_distance=mean(d);
    
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
    dist2=dot(average_distance,average_distance);
    
    if dist2<max_smooth2 && dist2>min_smooth2
        if average_distance(2)<0
            %disp('\n Smooth Pursuit Down!!!!');
            avg_type(k,:)=[average_distance,[0 1 0]];
            data_d(k:k+window_size-1,5)=1;
        elseif average_distance(2)>0
            %disp('\n Smooth Pursuit Up!!!!');
            avg_type(k,:)=[average_distance,[0 0 1]];
            data_d(k:k+window_size-1,5)=2;
        end
    elseif dist2 <min_smooth2
        %             if dist2>max_dist
        %                 max_dist=dist2;
        %             end
        %disp('\n Fixation!!!!');
        avg_type(k,:)=[average_distance,[1 0 0]];
        if data_d(k:k+window_size-1,5)~=1
            data_d(k:k+window_size-1,5)=3;
        end
    else
        %disp('\n Saccade? or whatever');
        avg_type(k,:)=[average_distance,[0 0 0]];
    end
    
    % end
    k=k+1;
end

%Plot average distance distribution
figure
scatter(avg_type(:,1),avg_type(:,2),1,avg_type(:,3:5));
hold on
r=sqrt(max_smooth2);
x_c=r*cos(teta);
y_c=r*sin(teta);
plot(x_c,y_c);

r=sqrt(min_smooth2);
x_c=r*cos(teta);
y_c=r*sin(teta);
plot(x_c,y_c);

r=sqrt(0.0064);
x_c=r*cos(teta);
y_c=r*sin(teta);
plot(x_c,y_c);
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
if exist('saccade_log')
    assignin('base', 'saccade_l', saccade_log)
end


end