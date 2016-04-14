function [ type, percentage ] = get_sequencetype( subject, video, timestamp_start, timestamp_end  )
% determines the type of sequence (SPU/SPD/FIX/SHAUN) based on the
% timestamp of the end of the sequence
% subject:      1-9
% vid:          1-2

% type:         1-4 (1=SPU, 2=SPD, 3=FIX, 4=SHAUN)
% percentage:   returns a percentage indicating how much of the sequence is
%               classified as type



%% CONSTANTS
NUM_CLUSTER = 8;

DURATION_CLUSTER = 4.067; 
DURATION_FIX = 2.533;
DURATION_SHAUN = 30.033;

DURATION_EPISODE = NUM_CLUSTER*DURATION_CLUSTER;
DURATION_E_S = DURATION_EPISODE + DURATION_SHAUN;

% Permutation of each episode 1 = SPU; 0 = SPD
    perm=[1 1 1 0 0 0 0 1;...
          1 1 0 1 0 1 0 0;...
          0 0 1 0 0 1 1 1;...
          1 1 0 0 1 0 1 0;...
          0 1 1 1 0 0 1 0;...
          1 1 0 1 0 0 1 0;...
          0 1 1 0 1 0 0 1;...
          0 1 1 1 0 0 0 1;...
          0 1 1 0 1 0 1 0;...
          0 1 0 1 0 0 1 1;...
          0 1 0 1 0 1 0 1;...
          1 1 0 1 0 1 0 0];


time_in_episode_end = mod(timestamp_end,DURATION_E_S);
time_in_episode_start = mod(timestamp_start,DURATION_E_S);

if time_in_episode_end > DURATION_EPISODE
    % Shaun
    type = 4;
    
    % Calculate percentage in Shaun
    if time_in_episode_start > DURATION_EPISODE
        percentage = 1;
    else
        percentage = (time_in_episode_end-DURATION_EPISODE)/(timestamp_end - timestamp_start);
    end
else
    % FIX/SPU/SPD
    time_in_cluster_end = mod(time_in_episode_end,DURATION_CLUSTER);
    time_in_cluster_start = mod(time_in_episode_start,DURATION_CLUSTER);
    
    if time_in_cluster_end < DURATION_FIX
        % FIX
        type = 3;
        
        % Calculate percentage in FIX
        if time_in_cluster_start < time_in_cluster_end
            percentage = 1;
        else
            percentage = time_in_cluster_end/(timestamp_end - timestamp_start);
        end
    else
        % SPU/SPD
        akt_episode_end=floor(timestamp_end/DURATION_E_S);
        akt_cluster_end=floor(time_in_episode_end/DURATION_CLUSTER);
        
        if perm(mod(subject+(video-1)*6+akt_episode_end-1,12)+1,akt_cluster_end+1)
            % SPU
            type=1;
        else
            % SPD
            type=2;
        end
        
        % Calculate percentage in SPU/SPD        
        if time_in_cluster_start > DURATION_FIX
            percentage = 1;
        else
            percentage = (time_in_cluster_end - DURATION_FIX)/(timestamp_end - timestamp_start);
        end
    end
end

    
end

