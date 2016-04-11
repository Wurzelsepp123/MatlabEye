function [ data_sac_split ] = split_saccade( data, min_length )
% Function to cut off saccades and write each episode in extra cell

% data: input data with the following columns 
%       [time [s] coordinates [°] coordinates [px] velocity[°/s] is_saccade[0/1]] 

l=1;
i=1;

while i<size(data,1)
    if data(i,5)==0     % start of episode detected 
        j=i;
        
        while data(i,5)==0 && i<size(data,1) % search for the end of episode
           i=i+1;
        end
        
        if i-j > min_length     % save episodes
            data_sac_split{l}=data(j:i-1,:);
            l=l+1;
        end
    end
    i=i+1;
end

if ~exist('data_sac_split','var')
    data_sac_split = [];
end
end

