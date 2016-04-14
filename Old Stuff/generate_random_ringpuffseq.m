function [ data_train_ring, data_val_ring, Perm] = generate_random_ringpuffseq(data, sequencelength, skip_samples,filter_coeff,g_h,h,d,cutoff)
% generates ringpuffersequences with length = sequencelength for training
% and validation

% sequences are velocities in [px/s] calculated from the filtered coordinates [px]

% skip_samples determines how many samples are skipped for the next start
% of the ringpuffer

Perm=cellfun(@(x) randperm(length(x)),data,'UniformOutput',false);
data = cellfun(@(x,y) x(y),data,Perm,'UniformOutput',false);

data_train = cellfun(@(x) x(1:round(2/3*length(x))),data,'UniformOutput',false);
data_val = cellfun(@(x) x(round(2/3*length(x))+1:end),data,'UniformOutput',false);

cnt=0;
for i=1:size(data_train,1)
    for j=1:size(data_train,2)
        for k=1:length(data_train{i,j})
            for l=1:skip_samples:size(data_train{i,j}{k},1)-sequencelength
                cnt=cnt+1;
                tmp = conv(data_train{i,j}{k}(l:l+sequencelength-1,2)',filter_coeff);
                tmp = (1-2*d/h*tand(tmp))*g_h/2;
                if cutoff
                    data_train_ring{cnt}=diff(tmp((length(filter_coeff)+1)/2:end-(length(filter_coeff)-1)/2));
                else
                    data_train_ring{cnt}=diff(tmp);
                end
            end
        end
    end
end

cnt=0;
for i=1:size(data_val,1)
    for j=1:size(data_val,2)
        for k=1:length(data_val{i,j})
            for l=1:skip_samples:size(data_val{i,j}{k},1)-sequencelength
                cnt=cnt+1;
                tmp = conv(data_val{i,j}{k}(l:l+sequencelength-1,2)',filter_coeff);
                tmp = (1-2*d/h*tand(tmp))*g_h/2;
                if cutoff
                    data_val_ring{cnt}=diff(tmp((length(filter_coeff)+1)/2:end-(length(filter_coeff)-1)/2));
                else
                    data_val_ring{cnt}=diff(tmp);
                end
            end
        end
    end
end
       
end

