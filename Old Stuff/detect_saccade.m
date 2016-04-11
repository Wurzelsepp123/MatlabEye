function [ data ] = detect_saccade( data, v_min_saccade, v_max_fix, min_angle, max_skip)
% Function to calculate velocity and detect saccades based on 3 input parameters
% data: input data with the following columns 
%       [time [s] coordinates [°] coordinates [px]]
% velocities in [°/s] and angle in [°]

% Saccades will be marked with 1 in the 5th column
% To be marked as saccade the following criteria:
%   - at least 1 velocity has to be higher than v_min_saccade
%   - every velocity during the saccade has to be higher than v_max_fix
%     (otherwise its classified as fixation)
%   - max_skip samples lower than v_max_fix will be ignored

% Saccades can be splitt using the split_saccade function
% min_angle is used to distinguish  between saccades and catch-up saccades

mean_timestamp=mean(diff(data(:,1)));
data(2:end,4)=diff(data(:,2))./mean_timestamp;
data(2:end,5)=(abs(data(2:end,4))>v_max_fix)|(data(2:end,3)==0);
data(2:end,6)=(abs(data(2:end,4))>v_min_saccade)|(data(2:end,3)==0);

i=1;
while i<size(data,1) 
    sum_angle=0;
    is_saccade=0;
    skip=0;  
    j=i;
    
    is_not_fix=data(j,5);

    while is_not_fix==1 && j<size(data,1)
        
        if data(j,5)==1         % no fixation (but not necessarily a saccade)
            skip=0;
        elseif skip < max_skip  % fixation but can be ignored
            data(j,5)=1;
            skip = skip + 1;
        else                    % more than 2 fixation in a row (can't be ignored)
            break;
        end

        if data(j,6)==1 
            is_saccade=1;
        end
        
        sum_angle=sum_angle+abs(data(j,4));
        j = j + 1;
    end
    if is_saccade==0 || sum_angle < min_angle/mean_timestamp
        data(i:j,5:6)=0; 
    end
    i= j + 1;
end

end
