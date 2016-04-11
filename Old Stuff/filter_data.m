function [ data ] = filter_data( data,filterr,g_h,h,d)
% function to filter the data and transform to px

% data: input data with the following columns 
%       [time [s] coordinates [°] coordinates [px] velocity[°/s]]

% The filterr has to be constructed using designfilt() function

%    g_h     [px]: Gaze height in 
%    h       [mm]: Video height in 
%    d       [mm]: Distance eyetracker - display


% The filtered data in ° will be saved in 5th column
% The filtered and afterwards transformed data in px will be saved in 6th
% column

data(:,5)=filter(filterr,data(:,2));
data(:,6)=(1-2*d/h*tand(data(:,5)))*g_h/2;
    

end

