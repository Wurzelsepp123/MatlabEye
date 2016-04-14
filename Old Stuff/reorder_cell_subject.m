function [ data_split_reorder ] = reorder_cell_subject( data_split )
% Reorder the data such that in each row is 1 subject and in each column are
% all episodes of 1 video

for i=1:size(data_split,1)
    
    cnt = 0;
    for j=1:size(data_split,2)/2
        if iscell(data_split{i,j})
            for k=1:length(data_split{i,j})
                cnt=cnt+1;
                data_split_reorder{i,1}{cnt}=data_split{i,j}{k};
            end
        else
            cnt=cnt+1;
            data_split_reorder{i,1}{cnt}=data_split{i,j};
        end
    end
    
    cnt = 0;
    for j=size(data_split,2)/2+1:size(data_split,2)
        if iscell(data_split{i,j})
            for k=1:length(data_split{i,j})
                cnt=cnt+1;
                data_split_reorder{i,2}{cnt}=data_split{i,j}{k};
            end
        else
            cnt=cnt+1;
            data_split_reorder{i,2}{cnt}=data_split{i,j};
        end      
    end   
    
end

if ~exist('data_split_reorder','var')
    data_split_reorder = data_split;
end


end

